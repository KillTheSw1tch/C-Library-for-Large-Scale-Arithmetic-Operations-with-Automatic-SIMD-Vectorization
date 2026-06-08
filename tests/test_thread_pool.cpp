#include <simd_array/thread_pool.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using simd::ThreadPool;

// ----------------------------------------------------------------------------
// Construction.
// ----------------------------------------------------------------------------

TEST(ThreadPool, DefaultCtorCreatesAtLeastOneWorker) {
    ThreadPool pool;
    EXPECT_GE(pool.thread_count(), 1u);
}

TEST(ThreadPool, ExplicitWorkerCountIsRespected) {
    ThreadPool pool(4);
    EXPECT_EQ(pool.thread_count(), 4u);
}

// ----------------------------------------------------------------------------
// submit().
// ----------------------------------------------------------------------------

TEST(ThreadPool, SubmitReturnsValue) {
    ThreadPool pool(2);
    auto fut = pool.submit([] { return 42; });
    EXPECT_EQ(fut.get(), 42);
}

TEST(ThreadPool, SubmitVoidTask) {
    ThreadPool pool(2);
    std::atomic<int> counter{0};
    auto fut = pool.submit([&counter] { counter.fetch_add(1); });
    fut.get();
    EXPECT_EQ(counter.load(), 1);
}

TEST(ThreadPool, ManySubmitsAllExecuteWithCorrectResults) {
    ThreadPool pool(4);
    constexpr int N = 200;
    std::vector<std::future<int>> futures;
    futures.reserve(N);
    for (int i = 0; i < N; ++i) {
        futures.push_back(pool.submit([i] { return i * 2; }));
    }
    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(futures[i].get(), i * 2) << "i=" << i;
    }
}

TEST(ThreadPool, SubmitRunsOnAWorkerThread) {
    ThreadPool pool(2);
    auto fut = pool.submit([] { return std::this_thread::get_id(); });
    EXPECT_NE(fut.get(), std::this_thread::get_id());
}

TEST(ThreadPool, ExceptionPropagatesViaFuture) {
    ThreadPool pool(2);
    auto fut = pool.submit([] { throw std::runtime_error("boom"); });
    EXPECT_THROW(fut.get(), std::runtime_error);
}

// ----------------------------------------------------------------------------
// parallel_for() — coverage and edge cases.
// ----------------------------------------------------------------------------

TEST(ThreadPool, ParallelForCoversWholeRangeExactlyOnce) {
    ThreadPool pool(4);
    constexpr std::size_t N = 1000;
    std::vector<int> hits(N, 0);
    pool.parallel_for(N, std::size_t{32}, [&hits](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) ++hits[i];
    });
    for (std::size_t i = 0; i < N; ++i) {
        ASSERT_EQ(hits[i], 1) << "i=" << i;
    }
}

TEST(ThreadPool, ParallelForEmptyRangeDoesNothing) {
    ThreadPool pool(2);
    int call_count = 0;
    pool.parallel_for(0, std::size_t{10}, [&call_count](std::size_t, std::size_t) {
        ++call_count;
    });
    EXPECT_EQ(call_count, 0);
}

TEST(ThreadPool, ParallelForSingleChunkRunsInline) {
    ThreadPool pool(4);
    std::thread::id task_thread;
    pool.parallel_for(10, std::size_t{100}, [&task_thread](std::size_t, std::size_t) {
        task_thread = std::this_thread::get_id();
    });
    EXPECT_EQ(task_thread, std::this_thread::get_id())
        << "single chunk should run on the calling thread";
}

TEST(ThreadPool, ParallelForChunkSizeEqualToN) {
    ThreadPool pool(2);
    std::size_t total_seen = 0;
    pool.parallel_for(50, std::size_t{50}, [&total_seen](std::size_t b, std::size_t e) {
        total_seen += (e - b);
    });
    EXPECT_EQ(total_seen, 50u);
}

TEST(ThreadPool, ParallelForLastChunkIsShorter) {
    ThreadPool pool(4);
    constexpr std::size_t N = 100;
    constexpr std::size_t CHUNK = 30;
    std::vector<std::pair<std::size_t, std::size_t>> seen;
    std::mutex mtx;
    pool.parallel_for(N, CHUNK, [&seen, &mtx](std::size_t b, std::size_t e) {
        std::lock_guard<std::mutex> lk(mtx);
        seen.emplace_back(b, e);
    });
    EXPECT_EQ(seen.size(), 4u);  // ceil(100/30) = 4
    std::size_t total = 0;
    for (auto& [b, e] : seen) total += (e - b);
    EXPECT_EQ(total, N);
}

TEST(ThreadPool, ParallelForUsesMultipleWorkers) {
    ThreadPool pool(4);
    constexpr std::size_t N = 100;
    std::set<std::thread::id> ids;
    std::mutex mtx;
    // Sleep keeps each chunk in flight long enough that work spreads across
    // workers — without it the first worker can sometimes drain everything.
    pool.parallel_for(N, std::size_t{1}, [&ids, &mtx](std::size_t, std::size_t) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::lock_guard<std::mutex> lk(mtx);
        ids.insert(std::this_thread::get_id());
    });
    EXPECT_GE(ids.size(), 2u);
}

TEST(ThreadPool, ParallelForAutoChunkSizeCoversWholeRange) {
    ThreadPool pool(4);
    constexpr std::size_t N = 10000;
    std::atomic<std::size_t> total{0};
    pool.parallel_for(N, [&total](std::size_t b, std::size_t e) {
        total.fetch_add(e - b);
    });
    EXPECT_EQ(total.load(), N);
}

TEST(ThreadPool, ParallelForArithmeticIsCorrect) {
    ThreadPool pool(4);
    constexpr std::size_t N = 10000;
    std::vector<int> a(N);
    std::iota(a.begin(), a.end(), 0);
    std::vector<int> out(N);
    pool.parallel_for(N, std::size_t{128}, [&a, &out](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) out[i] = a[i] * 2;
    });
    for (std::size_t i = 0; i < N; ++i) {
        ASSERT_EQ(out[i], static_cast<int>(i) * 2) << "i=" << i;
    }
}

// ----------------------------------------------------------------------------
// Lifecycle.
// ----------------------------------------------------------------------------

TEST(ThreadPool, DefaultPoolIsStableSingleton) {
    auto& a = simd::default_pool();
    auto& b = simd::default_pool();
    EXPECT_EQ(&a, &b);
    EXPECT_GE(a.thread_count(), 1u);
}

TEST(ThreadPool, DestructorDrainsPendingTasks) {
    std::atomic<int> count{0};
    {
        ThreadPool pool(2);
        for (int i = 0; i < 20; ++i) {
            pool.submit([&count] {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                count.fetch_add(1);
            });
        }
        // Pool destructor here: stop_ is set, but workers must still pop and
        // run the queued tasks before joining.
    }
    EXPECT_EQ(count.load(), 20);
}

TEST(ThreadPool, DestructorJoinsWithoutHanging) {
    auto start = std::chrono::steady_clock::now();
    {
        ThreadPool pool(8);
        for (int i = 0; i < 50; ++i) {
            pool.submit([] {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            });
        }
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_LT(elapsed, std::chrono::seconds(5)) << "destructor took unreasonably long";
}
