//! The vector env's thread pool: persistent workers that wait for the next job spinning for a
//! while before they sleep, and a calling thread that works as worker 0.
//!
//! A vector step hands each thread a few microseconds of work per env; a general-purpose pool's
//! handoff (rayon's `broadcast` takes 10-25 us to start and join its threads) costs as much as
//! the work at small batches. Here a job starts with one atomic store, the caller takes a share
//! itself instead of waiting idle, and workers that were spinning pick the job up at once.
//!
//! This is the crate's one use of `unsafe`: a job is a closure that borrows the caller's stack
//! (the envs and the output buffers), which the persistent workers can't hold as a `'static`
//! reference. [`Pool::run`] erases the borrow's lifetime and makes it sound by not returning,
//! panic or not, until every worker is done with the job.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

/// How long an idle worker spins for the next job before it sleeps, by default. A vector env
/// stepped in a loop hands out its next job well within this (the policy's forward pass in
/// between included, for small batches); one left alone stops costing CPU soon after.
pub const DEFAULT_SPIN: Duration = Duration::from_micros(100);

/// The longest spin [`Pool::set_spin`] sets: past a few milliseconds a parked worker's wakeup
/// costs next to nothing beside the wait, and a spinning one holds a core all along.
pub const MAX_SPIN: Duration = Duration::from_millis(10);

/// A job: worker `k` calls `f(k)`.
struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync + 'a),
}

/// A value on a cache line of its own, so workers polling one field don't contend with
/// writes to another.
#[repr(align(128))]
struct Padded<T>(T);

struct Shared {
    /// Bumped once per job; a worker runs each generation once.
    generation: Padded<AtomicU64>,
    /// The current job (valid from its generation's bump until `remaining` reaches 0).
    job: AtomicPtr<Job<'static>>,
    /// Workers that have yet to finish the current job.
    remaining: Padded<AtomicUsize>,
    /// The first panic a worker caught in the current job.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    /// Spin time in nanoseconds.
    spin_ns: AtomicU64,
    quit: AtomicBool,
    /// Tests: make the workers panic while they wait for a job.
    #[cfg(test)]
    fail_idle: AtomicBool,
}

/// `threads` threads to run jobs on: the caller and `threads - 1` workers.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<(Thread, JoinHandle<()>)>,
}

impl Pool {
    /// A pool of `threads` threads (at least 1; the caller is one of them).
    pub fn new(threads: usize) -> std::io::Result<Pool> {
        let shared = Arc::new(Shared {
            generation: Padded(AtomicU64::new(0)),
            job: AtomicPtr::new(std::ptr::null_mut()),
            remaining: Padded(AtomicUsize::new(0)),
            panic: Mutex::new(None),
            spin_ns: AtomicU64::new(DEFAULT_SPIN.as_nanos() as u64),
            quit: AtomicBool::new(false),
            #[cfg(test)]
            fail_idle: AtomicBool::new(false),
        });
        let mut pool = Pool {
            shared,
            workers: Vec::new(),
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

    /// How long idle workers spin for the next job before they sleep, at most [`MAX_SPIN`].
    pub fn set_spin(&self, spin: Duration) {
        let ns = spin.min(MAX_SPIN).as_nanos() as u64;
        self.shared.spin_ns.store(ns, Ordering::Relaxed);
    }

    /// Call `f(k)` for every thread `k` in `0..threads()` at once, `f(0)` on this thread, and
    /// return when all have returned. A panic in any of them is resumed here, after all have
    /// returned.
    pub fn run(&mut self, f: &(dyn Fn(usize) + Sync)) {
        if self.workers.is_empty() {
            f(0);
            return;
        }
        let shared = &*self.shared;
        let job = Job { f };
        // SAFETY (the lifetime erasure): workers dereference this pointer only between the
        // generation bump below and their decrement of `remaining`, and this function doesn't
        // return or unwind before `remaining` is 0, so `job` and what `f` borrows outlive every
        // use. `&mut self` keeps a second job from replacing it meanwhile.
        let ptr = std::ptr::from_ref(&job).cast::<Job<'static>>().cast_mut();
        shared.job.store(ptr, Ordering::Relaxed);
        shared
            .remaining
            .0
            .store(self.workers.len(), Ordering::Relaxed);
        // Publishes the job, `remaining` and everything `f` borrows to the workers.
        shared.generation.0.fetch_add(1, Ordering::Release);
        for (t, _) in &self.workers {
            // An atomic swap unless the worker sleeps; a worker that gets a stale wakeup
            // just goes back to sleep.
            t.unpark();
        }
        let mine = catch_unwind(AssertUnwindSafe(|| f(0)));
        let mut spins = 0u32;
        while shared.remaining.0.load(Ordering::Acquire) != 0 {
            spins = spins.saturating_add(1);
            if spins < 1 << 14 {
                std::hint::spin_loop();
            } else {
                // More threads than free cores: let the workers run.
                thread::yield_now();
            }
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

/// Worker `k`: run [`work_loop`], aborting the process if it panics. Jobs' panics are caught
/// and handed to the caller; one outside a job would end the thread and leave the next
/// [`Pool::run`] waiting for it forever.
fn work(shared: &Shared, k: usize) {
    if catch_unwind(AssertUnwindSafe(|| work_loop(shared, k))).is_err() {
        std::process::abort();
    }
}

/// Worker `k`'s loop: wait for a job, run it, report it done.
fn work_loop(shared: &Shared, k: usize) {
    let mut seen = 0u64;
    loop {
        let mut idle_since: Option<Instant> = None;
        'wait: loop {
            // Poll a while between looks at the clock.
            for _ in 0..64 {
                let g = shared.generation.0.load(Ordering::Acquire);
                if g != seen {
                    seen = g;
                    break 'wait;
                }
                if shared.quit.load(Ordering::Acquire) {
                    return;
                }
                std::hint::spin_loop();
            }
            #[cfg(test)]
            assert!(
                !shared.fail_idle.load(Ordering::Relaxed),
                "worker {k} fails"
            );
            let since = *idle_since.get_or_insert_with(Instant::now);
            let spin = Duration::from_nanos(shared.spin_ns.load(Ordering::Relaxed));
            if since.elapsed() >= spin {
                // `Pool::run` and `drop` unpark every worker after their store, so a job or a
                // quit that lands between the checks above and this park ends the park at once.
                thread::park();
            }
        }
        let ptr = shared.job.load(Ordering::Relaxed);
        // SAFETY: the generation this worker just saw was bumped (Release, read with Acquire)
        // after `ptr` was stored, and `Pool::run` keeps the job alive until this worker
        // decrements `remaining` below.
        let job = unsafe { &*ptr };
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| (job.f)(k))) {
            let mut slot = shared.panic.lock().unwrap_or_else(PoisonError::into_inner);
            slot.get_or_insert(p);
        }
        // Last use of the job: after this, `Pool::run` may return and drop it.
        shared.remaining.0.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_thread_runs_every_job_once() {
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
                pool.run(&|k| {
                    counts[k].fetch_add(1, Ordering::Relaxed);
                });
            }
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 500));
        }
    }

    #[test]
    fn a_worker_panic_reaches_the_caller_after_the_job() {
        let mut pool = Pool::new(3).unwrap();
        let done = AtomicUsize::new(0);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(&|k| {
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
        pool.run(&|_| {
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
            let ran = AtomicUsize::new(0);
            pool.run(&|_| {
                ran.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(ran.load(Ordering::Relaxed), 3);
        }
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
