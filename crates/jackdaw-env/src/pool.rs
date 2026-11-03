//! The vector env's thread pool: persistent workers that wait for the next job spinning for a
//! while before they sleep, and a calling thread that works as worker 0.
//!
//! A vector step hands each thread a few microseconds of work per env; a general-purpose pool's
//! handoff (rayon's `broadcast` takes 10-25 us to start and join its threads) costs as much as
//! the work at small batches. Here a job starts with one atomic store, the caller takes a share
//! itself instead of waiting idle, and workers that were spinning pick the job up at once.
//!
//! Workers only help: the caller's `f(0)` must get the whole job done if no worker joins, and a
//! job closes when it returns. So a worker that is late, descheduled because the pool has more
//! threads than there are free cores, or asleep, holds nothing up; the caller waits only for
//! the workers that joined to finish. Idle workers spin briefly, then poll with `yield_now`,
//! handing their core to any thread that wants it, then sleep.
//!
//! This is the crate's one use of `unsafe`: a job is a closure that borrows the caller's stack
//! (the envs and the output buffers), which the persistent workers can't hold as a `'static`
//! reference. [`Pool::run`] erases the borrow's lifetime and makes it sound by not returning,
//! panic or not, until no worker can still use the job.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

/// How long an idle worker waits for the next job before it sleeps, by default. A vector env
/// stepped in a loop hands out its next job well within this (the policy's forward pass in
/// between included, for small batches); one left alone stops costing CPU soon after.
pub const DEFAULT_SPIN: Duration = Duration::from_micros(100);

/// The longest spin [`Pool::set_spin`] sets: past a few milliseconds a parked worker's wakeup
/// costs next to nothing beside the wait, and a spinning one holds a core all along.
pub const MAX_SPIN: Duration = Duration::from_millis(10);

/// How much of its spin an idle worker spends polling flat out; it polls with `yield_now`
/// between looks for the rest.
const BUSY_SPIN: Duration = Duration::from_micros(20);

/// How long the caller polls flat out for the workers to finish a job before it polls with
/// `yield_now` between looks.
const CALLER_SPIN: Duration = Duration::from_micros(20);

/// A gap this long between two looks at the clock in a flat-out poll means the thread was
/// descheduled: there are more runnable threads than cores, so yield from then on.
const PREEMPTED: Duration = Duration::from_micros(50);

/// A job: worker `k` calls `f(k)`.
struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync + 'a),
}

/// A value on a cache line of its own, so workers polling one field don't contend with
/// writes to another.
#[repr(align(128))]
struct Padded<T>(T);

struct Shared {
    /// The job workers may join: its generation (1 for the first job, bumped once per job), or
    /// 0 once the caller has closed it.
    open: Padded<AtomicU64>,
    /// The current job (valid while it is open, and until `active` reaches 0 after).
    job: AtomicPtr<Job<'static>>,
    /// Workers inside a job: joining it, running it or (late) turning away from it.
    active: Padded<AtomicUsize>,
    /// The first panic a worker caught in the current job.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    /// Spin time in nanoseconds.
    spin_ns: AtomicU64,
    quit: AtomicBool,
    /// Tests: make the workers panic while they wait for a job.
    #[cfg(test)]
    fail_idle: AtomicBool,
    /// Tests: worker `k` stalls this long before it joins each job.
    #[cfg(test)]
    stall: Mutex<Option<(usize, Duration)>>,
    /// Tests: how many times the workers have gone to sleep.
    #[cfg(test)]
    parks: AtomicUsize,
}

/// `threads` threads to run jobs on: the caller and `threads - 1` workers.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<(Thread, JoinHandle<()>)>,
    /// The last job's generation.
    generation: u64,
}

impl Pool {
    /// A pool of `threads` threads (at least 1; the caller is one of them).
    pub fn new(threads: usize) -> std::io::Result<Pool> {
        let shared = Arc::new(Shared {
            open: Padded(AtomicU64::new(0)),
            job: AtomicPtr::new(std::ptr::null_mut()),
            active: Padded(AtomicUsize::new(0)),
            panic: Mutex::new(None),
            spin_ns: AtomicU64::new(DEFAULT_SPIN.as_nanos() as u64),
            quit: AtomicBool::new(false),
            #[cfg(test)]
            fail_idle: AtomicBool::new(false),
            #[cfg(test)]
            stall: Mutex::new(None),
            #[cfg(test)]
            parks: AtomicUsize::new(0),
        });
        let mut pool = Pool {
            shared,
            workers: Vec::new(),
            generation: 0,
        };
        for k in 1..threads.max(1) {
            let shared = Arc::clone(&pool.shared);
            // On failure `pool` drops here, stopping the workers started so far.
            let handle = thread::Builder::new()
                .name(format!("jackdaw-vector-{k}"))
                .spawn(move || work(&shared, k))?;
            pool.workers.push((handle.thread().clone(), handle));
        }
        Ok(pool)
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// How long idle workers wait for the next job before they sleep, at most [`MAX_SPIN`].
    pub fn set_spin(&self, spin: Duration) {
        let ns = spin.min(MAX_SPIN).as_nanos() as u64;
        self.shared.spin_ns.store(ns, Ordering::Relaxed);
    }

    /// Call `f(0)` on this thread and `f(k)` on each worker `k` (in `1..threads()`) that joins
    /// the job before `f(0)` returns, and return when all those calls have returned. `f(0)`
    /// must do the whole job if need be: the workers only help. A panic in any of them is
    /// resumed here, after all have returned.
    pub fn run(&mut self, f: &(dyn Fn(usize) + Sync)) {
        if self.workers.is_empty() {
            f(0);
            return;
        }
        let shared = &*self.shared;
        let job = Job { f };
        // SAFETY (the lifetime erasure): a worker dereferences this pointer only after it has
        // counted itself in `active` and then seen the job still open, and stops using it
        // before it counts itself out. This function closes the job and then waits for
        // `active` to reach 0 before it returns or unwinds, so `job` and what `f` borrows
        // outlive every use. `&mut self` keeps a second job from replacing it meanwhile.
        let ptr = std::ptr::from_ref(&job).cast::<Job<'static>>().cast_mut();
        shared.job.store(ptr, Ordering::Relaxed);
        self.generation += 1;
        // Publishes the job and everything `f` borrows to the workers (SeqCst: see `join`).
        shared.open.0.store(self.generation, Ordering::SeqCst);
        for (t, _) in &self.workers {
            // An atomic swap unless the worker sleeps; a worker that gets a stale wakeup
            // just goes back to waiting.
            t.unpark();
        }
        let mine = catch_unwind(AssertUnwindSafe(|| f(0)));
        // Close the job: a worker that hasn't joined yet now turns away from it. Then wait
        // for the ones that did.
        shared.open.0.store(0, Ordering::SeqCst);
        if shared.active.0.load(Ordering::SeqCst) != 0 {
            wait_until(CALLER_SPIN, || shared.active.0.load(Ordering::Acquire) == 0);
        }
        shared.job.store(std::ptr::null_mut(), Ordering::Relaxed);
        let theirs = shared
            .panic
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Err(p) = mine {
            resume_unwind(p);
        }
        if let Some(p) = theirs {
            resume_unwind(p);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::Release);
        for (t, _) in &self.workers {
            t.unpark();
        }
        for (_, h) in self.workers.drain(..) {
            // A worker catches the jobs' panics, so it ends cleanly.
            let _ = h.join();
        }
    }
}

/// Poll `done` until it holds: flat out for `busy` (less once descheduled), then with
/// `yield_now` between looks, so a thread that shares this core gets it.
fn wait_until(busy: Duration, mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    let mut last = start;
    loop {
        // Poll a while between looks at the clock.
        for _ in 0..64 {
            if done() {
                return;
            }
            std::hint::spin_loop();
        }
        let now = Instant::now();
        if now - start >= busy || now - last >= PREEMPTED {
            break;
        }
        last = now;
    }
    while !done() {
        thread::yield_now();
    }
}

/// Worker `k`: run [`work_loop`], aborting the process if it panics. Jobs' panics are caught
/// and handed to the caller; one outside a job would end the thread and leave the next
/// [`Pool::run`] waiting for it forever.
fn work(shared: &Shared, k: usize) {
    if catch_unwind(AssertUnwindSafe(|| work_loop(shared, k))).is_err() {
        std::process::abort();
    }
}

/// Worker `k`'s loop: wait for a job, help with it, wait for the next.
fn work_loop(shared: &Shared, k: usize) {
    let mut seen = 0u64;
    while let Some(g) = wait_for_job(shared, seen) {
        seen = g;
        #[cfg(test)]
        if let Some((w, d)) = *shared.stall.lock().unwrap()
            && w == k
        {
            thread::sleep(d);
        }
        join(shared, k, g);
    }
}

/// Join job `g` and run `f(k)` if it is still open.
fn join(shared: &Shared, k: usize, g: u64) {
    // With SeqCst on both sides, either `Pool::run` sees this worker in `active` after it
    // closes the job (and waits for it), or this worker sees the job closed.
    shared.active.0.fetch_add(1, Ordering::SeqCst);
    if shared.open.0.load(Ordering::SeqCst) == g {
        let ptr = shared.job.load(Ordering::Relaxed);
        // SAFETY: `ptr` was stored before the job was opened (SeqCst, which releases, read
        // with SeqCst), and `Pool::run` keeps the job alive until this worker counts itself
        // out of `active` below.
        let job = unsafe { &*ptr };
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| (job.f)(k))) {
            let mut slot = shared.panic.lock().unwrap_or_else(PoisonError::into_inner);
            slot.get_or_insert(p);
        }
    }
    // Last use of the job: after this, `Pool::run` may return and drop it.
    shared.active.0.fetch_sub(1, Ordering::Release);
}

/// Wait for a job other than `seen` to open and return its generation, or `None` to quit:
/// poll flat out for a while, then with `yield_now` between looks, then sleep until
/// `Pool::run` or `drop` wakes this worker.
fn wait_for_job(shared: &Shared, seen: u64) -> Option<u64> {
    let poll = || {
        let g = shared.open.0.load(Ordering::Acquire);
        if g != 0 && g != seen {
            Some(Some(g))
        } else if shared.quit.load(Ordering::Acquire) {
            Some(None)
        } else {
            None
        }
    };
    loop {
        let spin = Duration::from_nanos(shared.spin_ns.load(Ordering::Relaxed));
        let start = Instant::now();
        let mut last = start;
        // Flat out.
        'busy: loop {
            for _ in 0..64 {
                if let Some(r) = poll() {
                    return r;
                }
                std::hint::spin_loop();
            }
            #[cfg(test)]
            assert!(
                !shared.fail_idle.load(Ordering::Relaxed),
                "worker fails while it waits"
            );
            let now = Instant::now();
            if now - start >= spin.min(BUSY_SPIN) || now - last >= PREEMPTED {
                break 'busy;
            }
            last = now;
        }
        // Yielding.
        while start.elapsed() < spin {
            if let Some(r) = poll() {
                return r;
            }
            thread::yield_now();
        }
        // Drop the wakeup left by a job this worker joined while it polled (`Pool::run` unparks
        // every worker), or the park below would return at once and the worker would poll for
        // a second spin. Any wakeup this takes is for a store that the poll after it sees.
        thread::park_timeout(Duration::ZERO);
        if let Some(r) = poll() {
            return r;
        }
        #[cfg(test)]
        shared.parks.fetch_add(1, Ordering::Relaxed);
        // `Pool::run` and `drop` unpark every worker after their store, so a job or a quit
        // that lands between the poll above and this park ends the park at once.
        thread::park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f(0)` that waits (up to a few seconds) for every worker to join before it returns.
    fn with_everyone(threads: usize, joined: &AtomicUsize, k: usize) {
        joined.fetch_add(1, Ordering::Relaxed);
        if k == 0 {
            let start = Instant::now();
            while joined.load(Ordering::Relaxed) < threads
                && start.elapsed() < Duration::from_secs(5)
            {
                thread::yield_now();
            }
        }
    }

    #[test]
    fn workers_join_jobs() {
        for threads in [1, 2, 4, 7] {
            let mut pool = Pool::new(threads).unwrap();
            assert_eq!(pool.threads(), threads);
            let counts: Vec<AtomicUsize> = (0..threads).map(|_| AtomicUsize::new(0)).collect();
            for round in 0..500 {
                if round == 250 {
                    // Let the workers fall asleep once.
                    pool.set_spin(Duration::ZERO);
                    thread::sleep(Duration::from_millis(5));
                }
                let joined = AtomicUsize::new(0);
                pool.run(&|k| {
                    with_everyone(threads, &joined, k);
                    counts[k].fetch_add(1, Ordering::Relaxed);
                });
                // Everyone who joined is done.
                assert_eq!(joined.load(Ordering::Relaxed), threads);
            }
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 500));
        }
    }

    #[test]
    fn every_item_is_done_once_whoever_joins() {
        for threads in [1, 2, 4, 7] {
            let mut pool = Pool::new(threads).unwrap();
            for round in 0..2000 {
                if round == 1000 {
                    pool.set_spin(Duration::ZERO);
                }
                let next = AtomicUsize::new(0);
                let done: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
                pool.run(&|_| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(d) = done.get(i) else { break };
                        d.fetch_add(1, Ordering::Relaxed);
                    }
                });
                assert!(done.iter().all(|d| d.load(Ordering::Relaxed) == 1));
            }
        }
    }

    #[test]
    fn a_late_worker_holds_nothing_up() {
        let mut pool = Pool::new(3).unwrap();
        *pool.shared.stall.lock().unwrap() = Some((2, Duration::from_secs(1)));
        let start = Instant::now();
        let ran = AtomicUsize::new(0);
        for _ in 0..100 {
            pool.run(&|_| {
                ran.fetch_add(1, Ordering::Relaxed);
            });
        }
        // Waiting for worker 2 would take a second.
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "{:?}",
            start.elapsed()
        );
        assert!(ran.load(Ordering::Relaxed) >= 100);
        *pool.shared.stall.lock().unwrap() = None;
    }

    #[test]
    fn a_worker_panic_reaches_the_caller_after_the_job() {
        let mut pool = Pool::new(3).unwrap();
        let done = AtomicUsize::new(0);
        let joined = AtomicUsize::new(0);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(&|k| {
                with_everyone(3, &joined, k);
                if k == 2 {
                    panic!("worker 2");
                }
                thread::sleep(Duration::from_millis(2));
                done.fetch_add(1, Ordering::Relaxed);
            });
        }));
        assert!(r.is_err());
        assert_eq!(done.load(Ordering::Relaxed), 2);
        // The pool still works.
        let ran = AtomicUsize::new(0);
        let joined = AtomicUsize::new(0);
        pool.run(&|k| {
            with_everyone(3, &joined, k);
            ran.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(ran.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn the_spin_is_capped() {
        let pool = Pool::new(2).unwrap();
        pool.set_spin(Duration::MAX);
        assert_eq!(
            pool.shared.spin_ns.load(Ordering::Relaxed),
            MAX_SPIN.as_nanos() as u64
        );
        pool.set_spin(Duration::from_micros(7));
        assert_eq!(pool.shared.spin_ns.load(Ordering::Relaxed), 7_000);
    }

    #[test]
    fn workers_keep_running_jobs_after_a_long_spin() {
        let mut pool = Pool::new(3).unwrap();
        pool.set_spin(Duration::MAX);
        for _ in 0..3 {
            thread::sleep(MAX_SPIN + Duration::from_millis(5));
            let joined = AtomicUsize::new(0);
            pool.run(&|k| with_everyone(3, &joined, k));
            assert_eq!(joined.load(Ordering::Relaxed), 3);
        }
    }

    #[test]
    fn idle_workers_spin_once_before_they_sleep() {
        // `run` wakes every worker, spinning ones too. A worker that joined a job while it spun
        // used to keep that wakeup, so its next park returned at once and it spun twice.
        let mut pool = Pool::new(4).unwrap();
        pool.set_spin(MAX_SPIN);
        let joined = AtomicUsize::new(0);
        pool.run(&|k| with_everyone(4, &joined, k));
        let before = pool.shared.parks.load(Ordering::Relaxed);
        thread::sleep(MAX_SPIN * 10);
        let parks = pool.shared.parks.load(Ordering::Relaxed) - before;
        // One sleep per worker (a spurious wakeup may add one); a second spin each makes 6.
        assert!(parks < 2 * 3, "{parks} parks for 3 workers after a job");
    }

    /// The child half of `a_worker_dying_outside_a_job_aborts`: run only in the child process.
    #[test]
    fn worker_dies_outside_a_job() {
        if std::env::var_os("JACKDAW_POOL_DEATH_CHILD").is_none() {
            return;
        }
        let mut pool = Pool::new(2).unwrap();
        pool.shared.fail_idle.store(true, Ordering::Relaxed);
        thread::sleep(Duration::from_millis(50));
        // The worker is dead (or the process with it): this must not hang.
        pool.run(&|_| {});
        thread::sleep(Duration::from_secs(5));
    }

    #[test]
    fn a_worker_dying_outside_a_job_aborts() {
        let exe = std::env::current_exe().unwrap();
        let mut child = std::process::Command::new(exe)
            .args([
                "--exact",
                "pool::tests::worker_dies_outside_a_job",
                "--test-threads=1",
            ])
            .env("JACKDAW_POOL_DEATH_CHILD", "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let start = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if start.elapsed() > Duration::from_secs(30) {
                child.kill().unwrap();
                panic!("the pool deadlocked after a worker died");
            }
            thread::sleep(Duration::from_millis(10));
        };
        assert!(!status.success());
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(status.signal(), Some(6), "SIGABRT: {status:?}");
        }
    }
}
