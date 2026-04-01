#pragma once
#include "gpu_device.h"
#include <vector>

enum class LinkType {
    NVLINK      = 100,  // Direct NVLink (~600 GB/s on H100)
    PCIE_PEER   = 50,   // Same PCIe switch (~32 GB/s)
    PCIE_CPU    = 10,   // Different PCIe switches, through CPU (~16 GB/s)
    UNKNOWN     = 1
};

class TopologyMatrix {
public:
    // Build from NVML: nvmlDeviceGetTopologyCommonAncestor for each pair
    void build(const std::vector<GPUDevice>& devices);

    // Get link type between two GPUs
    LinkType get_link(int gpu_a, int gpu_b) const;

    // Score a group of GPUs: sum of all pairwise link values
    int score_group(const std::vector<int>& gpu_ids) const;

    // Find best-connected group of N GPUs from candidates
    // Brute-force combination search (fine for N <= 8)
    std::vector<int> find_best_group(const std::vector<int>& candidates, int count) const;

    // Print topology matrix (matches nvidia-smi topo -m)
    void print() const;

    int gpu_count() const { return size_; }

private:
    std::vector<std::vector<LinkType>> matrix_;
    int size_ = 0;
};
