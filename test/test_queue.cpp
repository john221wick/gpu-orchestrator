// test_queue: Thread-safety and priority ordering tests for JobQueue
//
// Build: g++ -std=c++17 -Iinclude test/test_queue.cpp src/job_queue.cpp src/logger.cpp -o test_queue -lpthread
// Run:   ./test_queue

#include "job_queue.h"
#include "logger.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <algorithm>

static int tests_run = 0, tests_passed = 0;

#define TEST(name) \
    do { tests_run++; printf("  %-45s ", name); fflush(stdout); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return; } while(0)
#define ASSERT_TRUE(x) \
    do { if (!(x)) { printf("FAIL at line %d\n", __LINE__); return; } } while(0)
#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { printf("FAIL: expected %d got %d at line %d\n", (int)(b), (int)(a), __LINE__); return; } } while(0)

static JobRequest make_job(const std::string& id, int priority, time_t submitted = 0) {
    JobRequest j;
    j.job_id       = id;
    j.priority     = priority;
    j.submitted_at = submitted ? submitted : time(nullptr);
    return j;
}

// Test: try_pop on empty queue returns nullopt
static void test_try_pop_empty() {
    TEST("try_pop on empty queue returns nullopt");
    JobQueue q;
    auto result = q.try_pop();
    ASSERT_TRUE(!result.has_value());
    PASS();
}

// Test: size() and empty() are correct
static void test_size_empty() {
    TEST("size() and empty() track correctly");
    JobQueue q;
    ASSERT_TRUE(q.empty());
    ASSERT_EQ((int)q.size(), 0);

    q.push(make_job("j1", 5));
    ASSERT_TRUE(!q.empty());
    ASSERT_EQ((int)q.size(), 1);

    q.push(make_job("j2", 3));
    ASSERT_EQ((int)q.size(), 2);

    q.try_pop();
    ASSERT_EQ((int)q.size(), 1);

    q.try_pop();
    ASSERT_TRUE(q.empty());
    PASS();
}

// Test: Priority ordering — lower number = higher priority
static void test_priority_order() {
    TEST("Lower priority number dequeued first");
    JobQueue q;
    q.push(make_job("low",    10, 1));
    q.push(make_job("medium", 5,  2));
    q.push(make_job("high",   1,  3));
    q.push(make_job("urgent", 0,  4));

    auto j1 = q.try_pop(); ASSERT_TRUE(j1.has_value()); ASSERT_EQ(j1->priority, 0);
    auto j2 = q.try_pop(); ASSERT_TRUE(j2.has_value()); ASSERT_EQ(j2->priority, 1);
    auto j3 = q.try_pop(); ASSERT_TRUE(j3.has_value()); ASSERT_EQ(j3->priority, 5);
    auto j4 = q.try_pop(); ASSERT_TRUE(j4.has_value()); ASSERT_EQ(j4->priority, 10);
    PASS();
}

// Test: Same priority — earlier submission time dequeued first
static void test_fifo_same_priority() {
    TEST("Same priority: earlier submit time dequeued first");
    JobQueue q;
    q.push(make_job("first",  5, 100));
    q.push(make_job("second", 5, 200));
    q.push(make_job("third",  5, 300));

    auto j1 = q.try_pop(); ASSERT_TRUE(j1.has_value()); ASSERT_EQ(j1->submitted_at, 100);
    auto j2 = q.try_pop(); ASSERT_TRUE(j2.has_value()); ASSERT_EQ(j2->submitted_at, 200);
    auto j3 = q.try_pop(); ASSERT_TRUE(j3.has_value()); ASSERT_EQ(j3->submitted_at, 300);
    PASS();
}

// Test: Mixed priority and time
static void test_mixed_priority() {
    TEST("Mixed priority and time ordering");
    JobQueue q;
    q.push(make_job("p5-t1", 5, 1));
    q.push(make_job("p1-t2", 1, 2));
    q.push(make_job("p5-t3", 5, 3));
    q.push(make_job("p1-t4", 1, 4));
    q.push(make_job("p0-t5", 0, 5));

    auto j1 = q.try_pop(); ASSERT_TRUE(j1.has_value()); ASSERT_EQ(j1->priority, 0);
    auto j2 = q.try_pop(); ASSERT_TRUE(j2.has_value()); ASSERT_EQ(j2->priority, 1);
    auto j3 = q.try_pop(); ASSERT_TRUE(j3.has_value()); ASSERT_EQ(j3->priority, 1);
    auto j4 = q.try_pop(); ASSERT_TRUE(j4.has_value()); ASSERT_EQ(j4->priority, 5);
    auto j5 = q.try_pop(); ASSERT_TRUE(j5.has_value()); ASSERT_EQ(j5->priority, 5);
    PASS();
}

// Test: Thread-safe concurrent push from multiple producers
static void test_concurrent_push() {
    TEST("Concurrent push from 8 threads (100 jobs each)");
    JobQueue q;
    const int THREADS = 8;
    const int JOBS_PER_THREAD = 100;
    const int TOTAL = THREADS * JOBS_PER_THREAD;

    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for (int t = 0; t < THREADS; t++) {
        threads.emplace_back([&q, t](){
            for (int i = 0; i < JOBS_PER_THREAD; i++) {
                std::string id = "t" + std::to_string(t) + "-j" + std::to_string(i);
                q.push(make_job(id, rand() % 10, time(nullptr)));
            }
        });
    }
    for (auto& th : threads) th.join();

    ASSERT_EQ((int)q.size(), TOTAL);
    PASS();
}

// Test: Producer-consumer under contention
static void test_producer_consumer() {
    TEST("Producer-consumer: all jobs received correctly");
    JobQueue q;
    const int N = 500;
    std::atomic<int> consumed{0};

    std::thread producer([&](){
        for (int i = 0; i < N; i++) {
            q.push(make_job("job-" + std::to_string(i), i % 5, i));
        }
    });

    std::thread consumer([&](){
        for (int i = 0; i < N; i++) {
            JobRequest job = q.pop();  // blocking
            (void)job;
            consumed++;
        }
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(consumed.load(), N);
    ASSERT_TRUE(q.empty());
    PASS();
}

// Test: Multiple consumers
static void test_multi_consumer() {
    TEST("4 producers × 4 consumers: no jobs lost");
    JobQueue q;
    const int PRODUCERS = 4;
    const int CONSUMERS = 4;
    const int JOBS_EACH = 50;
    const int TOTAL = PRODUCERS * JOBS_EACH;
    std::atomic<int> produced{0}, consumed{0};
    std::atomic<bool> done{false};

    std::vector<std::thread> producers, consumers;

    for (int p = 0; p < PRODUCERS; p++) {
        producers.emplace_back([&, p](){
            for (int i = 0; i < JOBS_EACH; i++) {
                q.push(make_job("p" + std::to_string(p) + "-" + std::to_string(i), 5, i));
                produced++;
            }
        });
    }

    for (int c = 0; c < CONSUMERS; c++) {
        consumers.emplace_back([&](){
            while (true) {
                auto opt = q.try_pop();
                if (opt) {
                    consumed++;
                } else if (done && q.empty()) {
                    break;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });
    }

    for (auto& p : producers) p.join();
    done = true;
    q.notify_all();  // wake sleeping consumers
    for (auto& c : consumers) c.join();

    ASSERT_EQ(produced.load(), TOTAL);
    ASSERT_EQ(consumed.load(), TOTAL);
    PASS();
}

// Test: snapshot returns all jobs without removing them
static void test_snapshot() {
    TEST("snapshot() returns jobs without removing them");
    JobQueue q;
    q.push(make_job("j1", 3, 1));
    q.push(make_job("j2", 1, 2));
    q.push(make_job("j3", 5, 3));

    auto snap = q.snapshot();
    ASSERT_EQ((int)snap.size(), 3);
    ASSERT_EQ((int)q.size(), 3);  // still 3 in queue
    PASS();
}

int main() {
    logger_init();
    logger_set_level(LogLevel::ERROR);

    printf("\n=== JobQueue Tests ===\n\n");

    test_try_pop_empty();
    test_size_empty();
    test_priority_order();
    test_fifo_same_priority();
    test_mixed_priority();
    test_concurrent_push();
    test_producer_consumer();
    test_multi_consumer();
    test_snapshot();

    printf("\n%d/%d tests passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
