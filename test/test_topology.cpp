// test_topology: Unit tests for topology scoring and best-group selection
//
// Build: g++ -std=c++17 -DUSE_MOCK_NVML -Iinclude test/test_topology.cpp src/topology.cpp src/logger.cpp -o test_topology -lpthread
// Run:   ./test_topology

#include "topology.h"
#include "logger.h"

#include <cassert>
#include <cstdio>
#include <algorithm>

// Build a mock device list using mock handles
static std::vector<GPUDevice> make_mock_devices(int count) {
    std::vector<GPUDevice> devs(count);
    for (int i = 0; i < count; i++) {
        devs[i].id     = i;
        devs[i].handle = (nvmlDevice_t)(uintptr_t)(i + 1);  // mock handle
        snprintf(devs[i].name, sizeof(devs[i].name), "MockGPU-%d", i);
    }
    return devs;
}

static int tests_run = 0, tests_passed = 0;

#define TEST(name) \
    do { tests_run++; \
         printf("  %-40s ", name); fflush(stdout); } while(0)

#define PASS() \
    do { tests_passed++; printf("PASS\n"); } while(0)

#define FAIL(msg) \
    do { printf("FAIL: %s\n", msg); } while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { \
        printf("FAIL: expected %d, got %d at line %d\n", (int)(b), (int)(a), __LINE__); \
        return; } \
    } while(0)

#define ASSERT_TRUE(x) \
    do { if (!(x)) { \
        printf("FAIL: assertion failed at line %d\n", __LINE__); \
        return; } \
    } while(0)

// Test: NVLink pairs get correct LinkType
static void test_nvlink_detection() {
    TEST("NVLink pair detected correctly");
    auto devs = make_mock_devices(4);
    TopologyMatrix topo;
    topo.build(devs);

    // Mock: GPU0<->GPU1 NVLink, GPU2<->GPU3 NVLink
    ASSERT_EQ((int)topo.get_link(0, 1), (int)LinkType::NVLINK);
    ASSERT_EQ((int)topo.get_link(2, 3), (int)LinkType::NVLINK);
    PASS();
}

// Test: Cross-pair is PCIe-CPU
static void test_pcie_cpu_detection() {
    TEST("PCIe-CPU pair detected correctly");
    auto devs = make_mock_devices(4);
    TopologyMatrix topo;
    topo.build(devs);

    ASSERT_EQ((int)topo.get_link(0, 2), (int)LinkType::PCIE_CPU);
    ASSERT_EQ((int)topo.get_link(1, 3), (int)LinkType::PCIE_CPU);
    PASS();
}

// Test: Symmetric topology
static void test_symmetry() {
    TEST("Topology is symmetric");
    auto devs = make_mock_devices(4);
    TopologyMatrix topo;
    topo.build(devs);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i != j) {
                ASSERT_EQ((int)topo.get_link(i, j), (int)topo.get_link(j, i));
            }
        }
    }
    PASS();
}

// Test: Group scoring — NVLink pair should outscore PCIe pair
static void test_group_scoring() {
    TEST("NVLink group scores higher than PCIe group");
    auto devs = make_mock_devices(4);
    TopologyMatrix topo;
    topo.build(devs);

    int nvlink_score = topo.score_group({0, 1});   // NVLink
    int pcie_score   = topo.score_group({0, 2});   // PCIe-CPU

    ASSERT_TRUE(nvlink_score > pcie_score);
    ASSERT_EQ(nvlink_score, (int)LinkType::NVLINK);
    ASSERT_EQ(pcie_score, (int)LinkType::PCIE_CPU);
    PASS();
}

// Test: 4-GPU group score is sum of all 6 pairs
static void test_four_gpu_score() {
    TEST("4-GPU group score = sum of all 6 pairs");
    auto devs = make_mock_devices(4);
    TopologyMatrix topo;
    topo.build(devs);

    int score = topo.score_group({0, 1, 2, 3});
    // Pairs: 0-1=NVLink(100), 0-2=PCIe-CPU(10), 0-3=PCIe-CPU(10),
    //        1-2=PCIe-CPU(10), 1-3=PCIe-CPU(10), 2-3=NVLink(100)
    // Total = 240
    ASSERT_EQ(score, 240);
    PASS();
}

// Test: find_best_group picks NVLink pair
static void test_best_group_2gpu() {
    TEST("find_best_group picks NVLink pair for 2 GPUs");
    auto devs = make_mock_devices(4);
    TopologyMatrix topo;
    topo.build(devs);

    // All 4 GPUs available, want best 2
    auto best = topo.find_best_group({0, 1, 2, 3}, 2);
    ASSERT_EQ((int)best.size(), 2);

    // The best pair should have NVLink (score=100)
    int score = topo.score_group(best);
    ASSERT_EQ(score, (int)LinkType::NVLINK);
    PASS();
}

// Test: find_best_group with limited candidates
static void test_best_group_limited() {
    TEST("find_best_group with mixed candidates");
    auto devs = make_mock_devices(4);
    TopologyMatrix topo;
    topo.build(devs);

    // Only GPUs 0,2,3 available — best 2 should be 2,3 (NVLink)
    auto best = topo.find_best_group({0, 2, 3}, 2);
    ASSERT_EQ((int)best.size(), 2);
    int score = topo.score_group(best);
    ASSERT_EQ(score, (int)LinkType::NVLINK);
    PASS();
}

// Test: find_best_group returns empty when not enough candidates
static void test_best_group_insufficient() {
    TEST("find_best_group returns empty when insufficient candidates");
    auto devs = make_mock_devices(4);
    TopologyMatrix topo;
    topo.build(devs);

    auto best = topo.find_best_group({0, 1}, 4);
    ASSERT_EQ((int)best.size(), 0);
    PASS();
}

// Test: find_best_group exact count returns same list
static void test_best_group_exact() {
    TEST("find_best_group with exact count returns input");
    auto devs = make_mock_devices(4);
    TopologyMatrix topo;
    topo.build(devs);

    auto best = topo.find_best_group({0, 1}, 2);
    ASSERT_EQ((int)best.size(), 2);
    PASS();
}

int main() {
    logger_init();
    logger_set_level(LogLevel::ERROR);  // suppress info logs during tests

    printf("\n=== Topology Tests ===\n\n");

    test_nvlink_detection();
    test_pcie_cpu_detection();
    test_symmetry();
    test_group_scoring();
    test_four_gpu_score();
    test_best_group_2gpu();
    test_best_group_limited();
    test_best_group_insufficient();
    test_best_group_exact();

    printf("\n%d/%d tests passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
