//! A sampling profile of the engine's hot paths, with no profiler to install and no admin rights
//! (Windows, MSVC builds: the symbols come from the PDB).
//!
//! ```text
//! cargo run --release --example profile -- [WORKLOAD] [SECONDS] [FOLDED_OUT]
//! ```
//!
//! WORKLOAD is `env` (the Gym loop: observation, mask, a random legal action, the step; the
//! default), `new_run`, `play` (a 5-card play on a copy of a fresh round) or `shop` (cash out,
//! reroll and leave a shop). A sampler thread suspends the workload thread every millisecond,
//! reads its registers and walks its stack with `RtlVirtualUnwind` (nothing is allocated while
//! the thread is stopped); the return addresses are named with DbgHelp afterwards. The report
//! lists the functions by self time and by total time; FOLDED_OUT also gets the stacks in the
//! folded format flame-graph tools read. Inlined functions count as their caller.

#[cfg(all(windows, target_arch = "x86_64"))]
fn main() {
    sampler::main();
}

#[cfg(not(all(windows, target_arch = "x86_64")))]
fn main() {
    eprintln!("this profiler is for Windows x64 (MSVC); elsewhere use perf or samply");
}

#[cfg(all(windows, target_arch = "x86_64"))]
mod sampler {
    use std::collections::{HashMap, HashSet};
    use std::ffi::c_void;
    use std::hint::black_box;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use jackdaw_core::action::Action;
    use jackdaw_core::game::Game;
    use jackdaw_core::run::RunSpec;
    use jackdaw_env::env::{EnvConfig, GYM_OBS_LEN, GymEnv};
    use jackdaw_env::gym::MAX_ACTIONS;

    /// The x64 `CONTEXT` (winnt.h), as raw bytes: only `ContextFlags`, `Rsp` and `Rip` are read
    /// here; `RtlVirtualUnwind` updates the rest.
    #[repr(C, align(16))]
    struct Context([u8; 1232]);

    const CONTEXT_FLAGS: usize = 0x30;
    const RSP: usize = 0x98;
    const RIP: usize = 0xF8;
    /// `CONTEXT_FULL` for AMD64: control, integer and floating-point registers.
    const CONTEXT_FULL: u32 = 0x0010_000B;

    impl Context {
        fn get(&self, at: usize) -> u64 {
            u64::from_le_bytes(self.0[at..at + 8].try_into().unwrap())
        }

        fn set(&mut self, at: usize, v: u64) {
            self.0[at..at + 8].copy_from_slice(&v.to_le_bytes());
        }
    }

    /// `SYMBOL_INFOW` with room for a 1024-character name.
    #[repr(C)]
    struct SymbolInfo {
        size_of_struct: u32,
        type_index: u32,
        reserved: [u64; 2],
        index: u32,
        size: u32,
        mod_base: u64,
        flags: u32,
        value: u64,
        address: u64,
        register: u32,
        scope: u32,
        tag: u32,
        name_len: u32,
        max_name_len: u32,
        name: [u16; 1024],
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThreadId() -> u32;
        fn GetCurrentProcess() -> isize;
        fn OpenThread(access: u32, inherit: i32, thread_id: u32) -> isize;
        fn CloseHandle(handle: isize) -> i32;
        fn SuspendThread(thread: isize) -> u32;
        fn ResumeThread(thread: isize) -> u32;
        fn GetThreadContext(thread: isize, context: *mut Context) -> i32;
        fn RtlLookupFunctionEntry(
            pc: u64,
            image_base: *mut u64,
            history: *mut c_void,
        ) -> *mut c_void;
        fn RtlVirtualUnwind(
            handler_type: u32,
            image_base: u64,
            pc: u64,
            entry: *mut c_void,
            context: *mut Context,
            handler_data: *mut *mut c_void,
            establisher_frame: *mut u64,
            context_pointers: *mut c_void,
        ) -> *mut c_void;
    }

    /// `IMAGEHLP_LINEW64`.
    #[repr(C)]
    struct LineInfo {
        size_of_struct: u32,
        key: *mut c_void,
        line_number: u32,
        file_name: *const u16,
        address: u64,
    }

    #[link(name = "dbghelp")]
    unsafe extern "system" {
        fn SymSetOptions(options: u32) -> u32;
        fn SymInitializeW(process: isize, search_path: *const u16, invade: i32) -> i32;
        fn SymFromAddrW(
            process: isize,
            address: u64,
            displacement: *mut u64,
            symbol: *mut SymbolInfo,
        ) -> i32;
        fn SymGetLineFromAddrW64(
            process: isize,
            address: u64,
            displacement: *mut u32,
            line: *mut LineInfo,
        ) -> i32;
    }

    const THREAD_ACCESS: u32 = 0x0002 | 0x0008 | 0x0040; // suspend/resume, get context, query
    const MAX_FRAMES: usize = 96;

    /// One stack: the return addresses, innermost first.
    type Stack = Vec<u64>;

    /// Walk the stopped thread's stack from its registers. Reads memory and the images' unwind
    /// tables only: safe while the thread is suspended (it may hold the heap lock).
    unsafe fn walk(thread: isize, frames: &mut [u64; MAX_FRAMES]) -> usize {
        let mut ctx = Context([0; 1232]);
        ctx.0[CONTEXT_FLAGS..CONTEXT_FLAGS + 4].copy_from_slice(&CONTEXT_FULL.to_le_bytes());
        if unsafe { GetThreadContext(thread, &mut ctx) } == 0 {
            return 0;
        }
        let mut n = 0;
        while n < MAX_FRAMES {
            let pc = ctx.get(RIP);
            if pc == 0 {
                break;
            }
            frames[n] = pc;
            n += 1;
            let mut base = 0u64;
            let entry = unsafe { RtlLookupFunctionEntry(pc, &mut base, std::ptr::null_mut()) };
            if entry.is_null() {
                // A leaf function: the return address is on top of the stack.
                let rsp = ctx.get(RSP);
                if rsp == 0 {
                    break;
                }
                let ret = unsafe { *(rsp as *const u64) };
                ctx.set(RIP, ret);
                ctx.set(RSP, rsp + 8);
            } else {
                let mut handler_data = std::ptr::null_mut();
                let mut frame = 0u64;
                unsafe {
                    RtlVirtualUnwind(
                        0, // UNW_FLAG_NHANDLER
                        base,
                        pc,
                        entry,
                        &mut ctx,
                        &mut handler_data,
                        &mut frame,
                        std::ptr::null_mut(),
                    );
                }
            }
        }
        n
    }

    fn sample(thread_id: u32, stop: &AtomicBool) -> Vec<Stack> {
        let thread = unsafe { OpenThread(THREAD_ACCESS, 0, thread_id) };
        assert!(thread != 0, "OpenThread failed");
        let mut samples = Vec::with_capacity(100_000);
        let mut frames = [0u64; MAX_FRAMES];
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(1));
            if unsafe { SuspendThread(thread) } == u32::MAX {
                break;
            }
            let n = unsafe { walk(thread, &mut frames) };
            unsafe { ResumeThread(thread) };
            if n > 0 {
                samples.push(frames[..n].to_vec());
            }
        }
        unsafe { CloseHandle(thread) };
        samples
    }

    /// Function start and name for an address (DbgHelp, after the run).
    struct Symbols {
        process: isize,
        cache: HashMap<u64, (u64, String)>,
    }

    impl Symbols {
        fn new() -> Symbols {
            let process = unsafe { GetCurrentProcess() };
            // Cargo links the example in deps/ and copies it (and its PDB) up: search beside it.
            let dir = std::env::current_exe().unwrap();
            let dir = dir.parent().unwrap().to_string_lossy().to_string();
            let path: Vec<u16> = dir.encode_utf16().chain([0]).collect();
            unsafe {
                // SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES
                SymSetOptions(0x2 | 0x4 | 0x10);
                SymInitializeW(process, path.as_ptr(), 1);
            }
            Symbols {
                process,
                cache: HashMap::new(),
            }
        }

        /// `file:line` of the instruction before a return address (or at a sampled pc).
        fn line(&self, pc: u64) -> Option<String> {
            let mut info = LineInfo {
                size_of_struct: std::mem::size_of::<LineInfo>() as u32,
                key: std::ptr::null_mut(),
                line_number: 0,
                file_name: std::ptr::null(),
                address: 0,
            };
            let mut displacement = 0u32;
            let ok =
                unsafe { SymGetLineFromAddrW64(self.process, pc, &mut displacement, &mut info) };
            if ok == 0 || info.file_name.is_null() {
                return None;
            }
            let mut len = 0;
            while unsafe { *info.file_name.add(len) } != 0 {
                len += 1;
            }
            let file = String::from_utf16_lossy(unsafe {
                std::slice::from_raw_parts(info.file_name, len)
            });
            let short = file.rsplit(['\\', '/']).take(2).collect::<Vec<_>>();
            Some(format!(
                "{}/{}:{}",
                short[1..].join("/"),
                short[0],
                info.line_number
            ))
        }

        fn lookup(&mut self, pc: u64) -> (u64, String) {
            if let Some(hit) = self.cache.get(&pc) {
                return hit.clone();
            }
            let mut info: Box<SymbolInfo> = Box::new(unsafe { std::mem::zeroed() });
            info.size_of_struct = 88; // sizeof(SYMBOL_INFOW)
            info.max_name_len = 1024;
            let mut displacement = 0u64;
            // A return address points after the call: look up the byte before it.
            let ok = unsafe { SymFromAddrW(self.process, pc - 1, &mut displacement, &mut *info) };
            let out = if ok != 0 {
                let len = (info.name_len as usize).min(1024);
                (info.address, String::from_utf16_lossy(&info.name[..len]))
            } else {
                (pc, format!("{pc:#x}"))
            };
            self.cache.insert(pc, out.clone());
            out
        }
    }

    fn spec(seed: &str) -> RunSpec {
        RunSpec {
            back: jackdaw_core::card::center_by_key("b_red").unwrap(),
            stake: 1,
            seed: seed.to_owned(),
            challenge: None,
        }
    }

    /// Run `workload` until `stop`; returns how many units it did.
    fn run(workload: &str, stop: &AtomicBool) -> u64 {
        let mut units = 0u64;
        match workload {
            "new_run" => {
                while !stop.load(Ordering::Relaxed) {
                    let _ = black_box(Game::new_run(&spec(&format!("P{:07}", units % 10_000_000))));
                    units += 1;
                }
            }
            "play" => {
                let mut round = Game::new_run(&spec("PROFILE1")).unwrap();
                round.step(&Action::SelectBlind).unwrap();
                while !stop.load(Ordering::Relaxed) {
                    let mut g = round.clone();
                    g.step_in_place(&Action::PlayHand(vec![0, 1, 2, 3, 4]))
                        .unwrap();
                    black_box(g);
                    units += 1;
                }
            }
            "play5j" | "win" => {
                let mut round = Game::new_run(&spec("PROFILE1")).unwrap();
                if workload == "play5j" {
                    for key in [
                        "j_joker",
                        "j_greedy_joker",
                        "j_lusty_joker",
                        "j_wrathful_joker",
                        "j_gluttenous_joker",
                    ] {
                        let c = jackdaw_core::card::center_by_key(key).unwrap();
                        round
                            .debug_add(&jackdaw_core::debug::AddSpec::new(
                                jackdaw_core::debug::AddKey::Joker(c),
                            ))
                            .unwrap();
                    }
                }
                round.step(&Action::SelectBlind).unwrap();
                round.gs.chips = if workload == "win" { 1e12 } else { -1e12 };
                while !stop.load(Ordering::Relaxed) {
                    let mut g = round.clone();
                    g.step_in_place(&Action::PlayHand(vec![0, 1, 2, 3, 4]))
                        .unwrap();
                    black_box(g);
                    units += 1;
                }
            }
            "cashout" | "booster" | "reroll" => {
                let mut eval = Game::new_run(&spec("PROFILE1")).unwrap();
                eval.step(&Action::SelectBlind).unwrap();
                eval.gs.chips = 1e12;
                eval.step(&Action::PlayHand(vec![0])).unwrap();
                let mut shop = eval.clone();
                shop.step(&Action::CashOut).unwrap();
                shop.gs.dollars = 1e12;
                while !stop.load(Ordering::Relaxed) {
                    match workload {
                        "cashout" => {
                            let mut g = eval.clone();
                            g.step_in_place(&Action::CashOut).unwrap();
                            black_box(g);
                        }
                        "booster" => {
                            let mut g = shop.clone();
                            g.step_in_place(&Action::OpenBooster(0)).unwrap();
                            black_box(g);
                        }
                        _ => shop.step_in_place(&Action::Reroll).unwrap(),
                    }
                    units += 1;
                }
            }
            "shop" => {
                let mut eval = Game::new_run(&spec("PROFILE1")).unwrap();
                eval.step(&Action::SelectBlind).unwrap();
                eval.gs.chips = 1e9;
                eval.step(&Action::PlayHand(vec![0])).unwrap();
                while !stop.load(Ordering::Relaxed) {
                    let mut g = eval.clone();
                    g.step_in_place(&Action::CashOut).unwrap();
                    g.gs.dollars = 100.0;
                    for _ in 0..3 {
                        g.step_in_place(&Action::Reroll).unwrap();
                    }
                    g.step_in_place(&Action::NextRound).unwrap();
                    black_box(g);
                    units += 1;
                }
            }
            _ => {
                let mut obs = vec![0f32; GYM_OBS_LEN];
                let mut mask = vec![0u8; MAX_ACTIONS];
                let mut env = GymEnv::new(&spec("PROFILE1"), EnvConfig::default(), 1).unwrap();
                let mut rng = 0x9E37_79B9_7F4A_7C15u64;
                let mut episode = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    env.write_obs(&mut obs);
                    env.write_action_mask(&mut mask);
                    black_box((&obs, &mask));
                    let n = env.table().len();
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let pick = (rng % n.max(1) as u64) as usize;
                    let _ = env.step(pick);
                    units += 1;
                    if env.ended() {
                        episode += 1;
                        env.reset(&spec(&format!("E{episode:07}")), Some(episode))
                            .unwrap();
                    }
                }
            }
        }
        units
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let workload = args.first().map_or("env", String::as_str).to_owned();
        let seconds: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10.0);
        let folded_out = args.get(2).cloned();

        let stop = Arc::new(AtomicBool::new(false));
        let main_id = unsafe { GetCurrentThreadId() };
        let sampler = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || sample(main_id, &stop))
        };
        let timer = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs_f64(seconds));
                stop.store(true, Ordering::Relaxed);
            })
        };
        let t0 = Instant::now();
        let units = run(&workload, &stop);
        let elapsed = t0.elapsed().as_secs_f64();
        timer.join().unwrap();
        let samples = sampler.join().unwrap();

        let mut symbols = Symbols::new();
        let mut self_time: HashMap<u64, u64> = HashMap::new();
        let mut total_time: HashMap<u64, u64> = HashMap::new();
        let mut names: HashMap<u64, String> = HashMap::new();
        let mut folded: HashMap<String, u64> = HashMap::new();
        let mut lines: HashMap<String, u64> = HashMap::new();
        for stack in &samples {
            // The sampled pc itself (not a return address): its own line.
            let at = symbols
                .line(stack[0])
                .unwrap_or_else(|| format!("{:#x}", stack[0]));
            *lines.entry(at).or_default() += 1;
            let funcs: Vec<u64> = stack
                .iter()
                .map(|&pc| {
                    let (start, name) = symbols.lookup(pc);
                    names.entry(start).or_insert(name);
                    start
                })
                .collect();
            *self_time.entry(funcs[0]).or_default() += 1;
            let mut seen = HashSet::new();
            for &f in &funcs {
                if seen.insert(f) {
                    *total_time.entry(f).or_default() += 1;
                }
            }
            if folded_out.is_some() {
                let line: Vec<&str> = funcs.iter().rev().map(|f| names[f].as_str()).collect();
                *folded.entry(line.join(";")).or_default() += 1;
            }
        }

        let total = samples.len().max(1) as f64;
        println!(
            "{workload}: {units} units in {elapsed:.1} s ({:.2} us each), {} samples",
            elapsed * 1e6 / units.max(1) as f64,
            samples.len()
        );
        for (title, table) in [("self", &self_time), ("total", &total_time)] {
            let mut rows: Vec<(u64, u64)> = table.iter().map(|(&f, &c)| (f, c)).collect();
            rows.sort_by_key(|r| std::cmp::Reverse(r.1));
            println!("\nby {title} time:");
            for (f, c) in rows.into_iter().take(45) {
                println!("{:6.2}%  {}", 100.0 * c as f64 / total, names[&f]);
            }
        }
        let mut rows: Vec<(&String, &u64)> = lines.iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(*r.1));
        println!("\nself time by source line (inlined code counts at its own line):");
        for (at, c) in rows.into_iter().take(40) {
            println!("{:6.2}%  {at}", 100.0 * *c as f64 / total);
        }
        if let Some(path) = folded_out {
            let mut lines: Vec<String> = folded.iter().map(|(k, v)| format!("{k} {v}")).collect();
            lines.sort();
            std::fs::write(&path, lines.join("\n") + "\n").unwrap();
            println!("\nfolded stacks: {path}");
        }
    }
}
