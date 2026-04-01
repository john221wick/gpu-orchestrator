#pragma once
#include "gpu_device.h"
#include "topology.h"
#include <vector>
#include <mutex>
#include <limits>
#include <unordered_map>

class GPUStateTracker {
public:
    // Initialize from discovered devices
    void init(std::vector<GPUDevice> devices);

    // Find free GPUs meeting requirements.
    // If needs_peer is true, uses topology to find best-connected group.
    std::vector<int> find_available(int count, size_t min_vram, bool needs_peer,
                                    const TopologyMatrix& topo);

    // Mark GPUs as busy (PID set to 0 initially, updated after fork)
    void mark_busy(const std::vector<int>& gpu_ids, pid_t pid, const std::string& job_id,
                   size_t reserved_vram = 0);

    // Update PID for a job after fork (job_id -> pid)
    void update_pid(const std::string& job_id, pid_t pid);

    // Free all GPUs owned by this PID
    void mark_free(pid_t pid);

    // Refresh VRAM info from NVML (call periodically)
    void refresh_vram();

    // Snapshot for status display
    std::vector<GPUDevice> snapshot() const;

    int free_count() const;
    int total_count() const;

private:
    std::vector<GPUDevice>              devices_;
    std::unordered_map<pid_t, std::vector<int>> pid_to_gpus_;
    std::unordered_map<std::string, pid_t>      jobid_to_pid_;
    mutable std::mutex                  mutex_;
};
