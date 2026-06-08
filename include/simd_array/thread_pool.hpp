#pragma once

/// \file thread_pool.hpp
/// Fixed-size thread pool used to execute SIMD kernels in parallel.
///
/// Design:
///   * `ThreadPool` owns a fixed set of worker threads spawned at
///     construction; they live until the pool is destroyed.
///   * Tasks are queued via a single mutex-guarded `std::queue` plus a
///     condition variable. Lock granularity is per-submit, which is fine
///     for our chunk size (one submit per O(thousands) of elements). A
///     lock-free queue would buy nothing here.
///   * `parallel_for(n, chunk_size, fn)` splits `[0, n)` into chunks of
///     size `chunk_size` (last possibly shorter), invokes
///     `fn(begin, end)` for each chunk on a worker, and blocks until
///     every chunk finishes. Exceptions raised inside `fn` are captured
///     per chunk and re-thrown when the originating future is waited on.
///   * `default_pool()` returns a process-wide singleton sized to
///     `std::thread::hardware_concurrency()` (clamped to >= 1). The
///     dispatcher in the future public Array<T> API will use it.
///
/// Concurrency contract:
///   * `fn` passed to `parallel_for` must be safe to invoke concurrently
///     with disjoint sub-ranges. The SIMD backend kernels satisfy this
///     trivially because they operate on contiguous, non-overlapping
///     output slices.
///   * Re-entering `parallel_for` from within a worker (nested parallel
///     regions) is unsupported — it can deadlock when every worker is
///     blocked on its own pending sub-tasks.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace simd {

class ThreadPool {
public:
    /// Construct with `num_threads` workers. `0` (the default) resolves to
    /// `std::thread::hardware_concurrency()`, clamped to at least 1 — some
    /// systems return 0 when the concurrency hint is unavailable.
    explicit ThreadPool(std::size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;
        }
        workers_.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(queue_mutex_);
            stop_ = true;
        }
        queue_cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&)                 = delete;
    ThreadPool& operator=(ThreadPool&&)      = delete;

    std::size_t thread_count() const noexcept { return workers_.size(); }

    /// Submit a callable for asynchronous execution. The returned future
    /// yields the result when ready, or rethrows any exception that
    /// escaped from `f`.
    template<typename F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<std::decay_t<F>>> {
        using R = std::invoke_result_t<std::decay_t<F>>;
        // packaged_task is move-only; wrap in shared_ptr so the type-erased
        // std::function in the queue stays copyable.
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto fut  = task->get_future();
        {
            std::lock_guard<std::mutex> lk(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool::submit on stopped pool");
            }
            queue_.emplace([task] { (*task)(); });
        }
        queue_cv_.notify_one();
        return fut;
    }

    /// Run `fn(begin, end)` over each chunk of `[0, n)` of size
    /// `chunk_size` (the last chunk may be shorter), blocking until every
    /// chunk completes. `fn` must be safe to invoke concurrently with
    /// disjoint ranges. A single-chunk problem (`chunk_size >= n`) runs
    /// inline on the calling thread to skip pool overhead.
    template<typename F>
    void parallel_for(std::size_t n, std::size_t chunk_size, F&& fn) {
        if (n == 0) return;
        if (chunk_size == 0 || chunk_size >= n) {
            fn(std::size_t{0}, n);
            return;
        }
        const std::size_t num_chunks = (n + chunk_size - 1) / chunk_size;
        std::vector<std::future<void>> futures;
        futures.reserve(num_chunks);
        for (std::size_t i = 0; i < num_chunks; ++i) {
            const std::size_t begin = i * chunk_size;
            const std::size_t end   = (begin + chunk_size < n) ? (begin + chunk_size) : n;
            // Capture `fn` by reference: its lifetime extends beyond every
            // future.get() below, so the workers always see a live object.
            futures.push_back(submit([begin, end, &fn] { fn(begin, end); }));
        }
        // Wait for all chunks; .get() also rethrows any captured exception.
        for (auto& f : futures) f.get();
    }

    /// Convenience overload that picks a chunk size automatically. Aims
    /// for ~4 chunks per worker so a slow chunk doesn't stall the whole
    /// reduction.
    template<typename F>
    void parallel_for(std::size_t n, F&& fn) {
        const std::size_t target = thread_count() * 4;
        std::size_t chunk = (target == 0) ? n : (n + target - 1) / target;
        if (chunk == 0) chunk = 1;
        parallel_for(n, chunk, std::forward<F>(fn));
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(queue_mutex_);
                queue_cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                // Drain the queue even after stop_ is set, so submitted
                // tasks always run before the pool is destroyed.
                if (stop_ && queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop();
            }
            task();
        }
    }

    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> queue_;
    std::mutex                        queue_mutex_;
    std::condition_variable           queue_cv_;
    bool                              stop_ = false;
};

/// Process-wide pool, lazily constructed on first call. Static-local
/// initialisation is thread-safe in C++11+.
inline ThreadPool& default_pool() {
    static ThreadPool pool;
    return pool;
}

} // namespace simd
