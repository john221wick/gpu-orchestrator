// test_scheduler: Unit tests for scheduling decisions using mock NVML
//
// Build: g++ -std=c++17 -DUSE_MOCK_NVML -Iinclude \
//        test/test_scheduler.cpp src/scheduler.cpp src/gpu_state.cpp \
//        src/topology.cpp src/job_queue.cpp src/launcher.cpp src/logger.cpp \
//        -o test_scheduler -lpthread
// Run:   ./test_scheduler

#include "scheduler.h"
#include "gpu_state.h"
#include "topology.h"
#include "job_queue.h"
#include "launcher.h"
#include "logger.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>
#include <atomic>

static int tests_run = 0, tests_passed = 0;

#define TEST(name) \
    do { tests_run++; printf("  %-55s ", name); fflush(stdout); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return; } while(0)
#define ASSERT_TRUE(x) \
    do { if (!(x)) { printf("FAIL at line %d\n", __LINE__); return; } } while(0)
#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { printf("FAIL: expected %d got %d at line %d\n", (int)(b), (int)(a), __LINE__); return; } } while(0)
#define ASSERT_STR_EQ(a, b) \
    do { if ((a) != (b)) { printf("FAIL: expected %s got %s at line %d\n", (b), (a).c_str(), __LINE__); return; } } while(0)

// Build 4 mock GPUs with topology
static std::vector<GPUDevice> make_mock_devices() {
    std::vector<GPUDevice> devs(4);
    for (int i = 0; i < 4; i++) {
        devs[i].id         = i;
        devs[i].handle     = (nvmlDevice_t)(uintptr_t)(i + 1);
        devs[i].vram_total = (size_t)40 * 1024 * 1024 * 1024ULL;
        devs[i].vram_free  = (size_t)40 * 1024 * 1024 * 1024ULL;
        devs[i].is_busy    = false;
        devs[i].owner_pid  = 0;
        snprintf(devs[i].name, sizeof(devs[i].name), "MockA100-%d", i);
    }
    return devs;
}

static JobRequest make_job(const std::string& id, int num_gpus, bool needs_peer,
                            size_t min_vram = 0, int priority = 5) {
    JobRequest j;
    j.job_id       = id;
    j.num_gpus     = num_gpus;
    j.needs_peer   = needs_peer;
    j.min_vram     = min_vram;
    j.priority     = priority;
    j.submitted_at = time(nullptr);
    j.framework    = Framework::PYTHON;
    j.script       = "echo hello";  // safe for testing
    return j;
}

// Dummy launcher that records launched jobs without actually fork/exec-ing
struct MockLauncher {
    std::vector<JobRequest> launched;
    void launch(JobRequest& job) {
        job.status = JobRequest::RUNNING;
        job.pids.push_back(99999);  // fake PID
        launched.push_back(job);
    }
};

// ---- Tests ----

// Test: Simple single-GPU job gets scheduled
static void test_single_gpu_scheduled() {
    TEST("Single-GPU job gets scheduled on first free GPU");

    auto devs = make_mock_devices();
    TopologyMatrix topo; topo.build(devs);
    GPUStateTracker state; state.init(devs);

    // find_available should return 1 GPU
    auto avail = state.find_available(1, 0, false, topo);
    ASSERT_EQ((int)avail.size(), 1);
    PASS();
}

// Test: Multi-GPU peer job picks NVLink pair
static void test_nvlink_peer_selection() {
    TEST("2-GPU peer job selects NVLink pair (0,1) not (0,2)");

    auto devs = make_mock_devices();
    TopologyMatrix topo; topo.build(devs);
    GPUStateTracker state; state.init(devs);

    auto avail = state.find_available(2, 0, true, topo);
    ASSERT_EQ((int)avail.size(), 2);

    // Should be {0,1} or {2,3} — both are NVLink
    int score = topo.score_group(avail);
    ASSERT_EQ(score, (int)LinkType::NVLINK);
    PASS();
}

// Test: mark_busy removes GPUs from available pool
static void test_mark_busy() {
    TEST("mark_busy removes GPUs from available pool");

    auto devs = make_mock_devices();
    TopologyMatrix topo; topo.build(devs);
    GPUStateTracker state; state.init(devs);

    // Mark GPU 0 and 1 busy
    state.mark_busy({0, 1}, 1001, "job-aaa");
    ASSERT_EQ(state.free_count(), 2);

    auto avail = state.find_available(1, 0, false, topo);
    ASSERT_EQ((int)avail.size(), 1);

    // Should be GPU 2 or 3
    ASSERT_TRUE(avail[0] == 2 || avail[0] == 3);
    PASS();
}

// Test: mark_free restores GPUs
static void test_mark_free() {
    TEST("mark_free restores GPUs to available pool");

    auto devs = make_mock_devices();
    TopologyMatrix topo; topo.build(devs);
    GPUStateTracker state; state.init(devs);

    state.mark_busy({0, 1}, 1001, "job-bbb");
    ASSERT_EQ(state.free_count(), 2);

    state.mark_free(1001);
    ASSERT_EQ(state.free_count(), 4);
    PASS();
}

// Test: VRAM filter excludes GPUs with insufficient free memory
static void test_vram_filter() {
    TEST("VRAM filter excludes GPUs below min_vram threshold");

    auto devs = make_mock_devices();
    // Set GPU 0 and 1 as low VRAM (simulate partially occupied)
    devs[0].vram_free = (size_t)5  * 1024 * 1024 * 1024ULL;  // 5 GB
    devs[1].vram_free = (size_t)5  * 1024 * 1024 * 1024ULL;
    devs[2].vram_free = (size_t)30 * 1024 * 1024 * 1024ULL;  // 30 GB
    devs[3].vram_free = (size_t)30 * 1024 * 1024 * 1024ULL;

    TopologyMatrix topo; topo.build(devs);
    GPUStateTracker state; state.init(devs);

    size_t min_vram = (size_t)20 * 1024 * 1024 * 1024ULL;  // need 20 GB
    auto avail = state.find_available(2, min_vram, false, topo);
    ASSERT_EQ((int)avail.size(), 2);

    // Should return GPUs 2 and 3 (only ones with >= 20 GB)
    ASSERT_TRUE(std::find(avail.begin(), avail.end(), 0) == avail.end());
    ASSERT_TRUE(std::find(avail.begin(), avail.end(), 1) == avail.end());
    PASS();
}

// Test: Request for more GPUs than available returns empty
static void test_insufficient_gpus() {
    TEST("Requesting more GPUs than available returns empty");

    auto devs = make_mock_devices();
    TopologyMatrix topo; topo.build(devs);
    GPUStateTracker state; state.init(devs);

    // Mark 3 GPUs busy, only 1 free
    state.mark_busy({0, 1, 2}, 1001, "job-ccc");
    ASSERT_EQ(state.free_count(), 1);

    auto avail = state.find_available(2, 0, false, topo);
    ASSERT_EQ((int)avail.size(), 0);  // can't fit 2 GPUs
    PASS();
}

// Test: Job goes to waiting queue when resources unavailable
static void test_job_queued_when_no_resources() {
    TEST("Job moves to waiting queue when no GPUs available");

    auto devs = make_mock_devices();
    TopologyMatrix topo; topo.build(devs);
    GPUStateTracker state; state.init(devs);

    // Occupy all GPUs
    state.mark_busy({0, 1, 2, 3}, 1001, "job-ddd");

    JobQueue pending, waiting;
    std::atomic<int> launch_count{0};

    Launcher launcher(state, [&](pid_t, const JobRequest&) { launch_count++; });
    Scheduler scheduler(pending, waiting, state, topo, launcher);

    // Submit job to pending
    auto job = make_job("job-eee", 1, false);
    pending.push(job);

    // Run scheduler briefly
    std::thread t([&]{ scheduler.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    scheduler.stop();
    t.join();

    // Job should be in waiting queue, not launched
    ASSERT_EQ(launch_count.load(), 0);
    ASSERT_EQ((int)waiting.size(), 1);
    PASS();
}

// Test: Waiting job gets scheduled when resources freed
static void test_waiting_job_rescheduled() {
    TEST("Waiting job is scheduled after resources are freed");

    auto devs = make_mock_devices();
    TopologyMatrix topo; topo.build(devs);
    GPUStateTracker state; state.init(devs);

    // Occupy all GPUs
    state.mark_busy({0, 1, 2, 3}, 1001, "job-fff");

    JobQueue pending, waiting;
    std::atomic<int> launch_count{0};

    Launcher launcher(state, [&](pid_t pid, const JobRequest& job) {
        launch_count++;
        // In real code this would be done via monitor; here we simulate
        state.update_pid(job.job_id, pid);
    });
    Scheduler scheduler(pending, waiting, state, topo, launcher);

    // Submit job to pending — will end up in waiting
    auto job = make_job("job-ggg", 1, false);
    pending.push(job);

    std::thread t([&]{ scheduler.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Job should be waiting
    ASSERT_EQ(launch_count.load(), 0);

    // Free all GPUs
    state.mark_free(1001);

    // Notify scheduler
    scheduler.on_job_complete("job-fff");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    scheduler.stop();
    t.join();

    // Job should now be launched
    ASSERT_EQ(launch_count.load(), 1);
    PASS();
}

// Test: Priority ordering — high-priority job runs before low-priority
static void test_priority_scheduling() {
    TEST("High-priority job runs before low-priority job");

    auto devs = make_mock_devices();  // 4 GPUs
    TopologyMatrix topo; topo.build(devs);
    GPUStateTracker state; state.init(devs);

    // Limit to 2 GPUs busy so only 2 can run
    state.mark_busy({2, 3}, 1001, "job-existing");

    JobQueue pending, waiting;
    std::vector<std::string> launch_order;
    std::mutex order_mutex;

    Launcher launcher(state, [&](pid_t pid, const JobRequest& job) {
        std::lock_guard<std::mutex> lock(order_mutex);
        launch_order.push_back(job.job_id);
        state.update_pid(job.job_id, pid);
    });
    Scheduler scheduler(pending, waiting, state, topo, launcher);

    // Push low-priority first, then high-priority
    pending.push(make_job("low-pri",  1, false, 0, 10));
    pending.push(make_job("high-pri", 1, false, 0, 1));

    std::thread t([&]{ scheduler.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    scheduler.stop();
    t.join();

    // high-pri should be launched (priority 1 before 10)
    ASSERT_TRUE(!launch_order.empty());
    ASSERT_STR_EQ(launch_order[0], "high-pri");
    PASS();
}

int main() {
    logger_init();
    logger_set_level(LogLevel::ERROR);

    printf("\n=== Scheduler Tests ===\n\n");

    test_single_gpu_scheduled();
    test_nvlink_peer_selection();
    test_mark_busy();
    test_mark_free();
    test_vram_filter();
    test_insufficient_gpus();
    test_job_queued_when_no_resources();
    test_waiting_job_rescheduled();
    test_priority_scheduling();

    printf("\n%d/%d tests passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
