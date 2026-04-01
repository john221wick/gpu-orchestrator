#include "gpu_state.h"
#include "logger.h"

#ifdef USE_MOCK_NVML
#  include "mock_nvml.h"
#else
#  include <nvml.h>
#endif

#include <algorithm>
#include <stdexcept>

void GPUStateTracker::init(std::vector<GPUDevice> devices) {
    std::lock_guard<std::mutex> lock(mutex_);
    devices_ = std::move(devices);
}

std::vector<int> GPUStateTracker::find_available(
    int count, size_t min_vram, bool needs_peer, const TopologyMatrix& topo)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Collect free GPUs with sufficient VRAM
    std::vector<int> candidates;
    for (const auto& dev : devices_) {
        if (!dev.is_busy && dev.vram_free >= min_vram) {
            candidates.push_back(dev.id);
        }
    }

    if ((int)candidates.size() < count) {
        return {};  // not enough free GPUs
    }

    if (count == 1) {
        // Best-fit policy: preserve larger GPUs for jobs that actually need them.
        int best_id = -1;
        size_t best_waste = std::numeric_limits<size_t>::max();
        for (int id : candidates) {
            size_t waste = devices_[id].vram_free - min_vram;
            if (waste < best_waste) {
                best_waste = waste;
                best_id = id;
            }
        }
        return (best_id >= 0) ? std::vector<int>{best_id} : std::vector<int>{};
    }

    if (!needs_peer) {
        // No topology preference — just return first N candidates
        candidates.resize(count);
        return candidates;
    }

    // Topology-aware: find best-connected group
    return topo.find_best_group(candidates, count);
}

void GPUStateTracker::mark_busy(
    const std::vector<int>& gpu_ids, pid_t pid, const std::string& job_id,
    size_t reserved_vram)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (int id : gpu_ids) {
        if (id >= 0 && id < (int)devices_.size()) {
            devices_[id].is_busy      = true;
            devices_[id].owner_pid    = pid;
            devices_[id].owner_job_id = job_id;
#ifdef USE_MOCK_NVML
            size_t used_vram = reserved_vram;
            if (used_vram == 0) {
                used_vram = (size_t)4 * 1024 * 1024 * 1024ULL;
            }
            if (used_vram >= devices_[id].vram_total) {
                devices_[id].vram_free = 0;
            } else {
                devices_[id].vram_free = devices_[id].vram_total - used_vram;
            }
#endif
        }
    }
    if (pid != 0) {
        pid_to_gpus_[pid] = gpu_ids;
    }
    jobid_to_pid_[job_id] = pid;
}

void GPUStateTracker::update_pid(const std::string& job_id, pid_t pid) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobid_to_pid_.find(job_id);
    if (it != jobid_to_pid_.end()) {
        pid_t old_pid = it->second;
        it->second = pid;

        std::vector<int> gpus;

        // Re-map GPU ownership from a previously tracked PID if one exists.
        auto git = pid_to_gpus_.find(old_pid);
        if (git != pid_to_gpus_.end()) {
            gpus = git->second;
            pid_to_gpus_.erase(git);
        } else {
            // Initial scheduling marks GPUs busy with pid=0 before fork/exec.
            // Recover those assignments by scanning the devices owned by job_id.
            for (const auto& dev : devices_) {
                if (dev.is_busy && dev.owner_job_id == job_id) {
                    gpus.push_back(dev.id);
                }
            }
        }

        if (!gpus.empty()) {
            pid_to_gpus_[pid] = gpus;
            for (int id : gpus) {
                if (id >= 0 && id < (int)devices_.size()) {
                    devices_[id].owner_pid = pid;
                }
            }
        }
    }
}

void GPUStateTracker::mark_free(pid_t pid) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pid_to_gpus_.find(pid);
    if (it == pid_to_gpus_.end()) return;

    for (int id : it->second) {
        if (id >= 0 && id < (int)devices_.size()) {
            LOG_INFO("FREE", "GPU %d released (was owned by PID %d, job %s)",
                     id, pid, devices_[id].owner_job_id.c_str());
            devices_[id].is_busy      = false;
            devices_[id].owner_pid    = 0;
            devices_[id].owner_job_id = "";
#ifdef USE_MOCK_NVML
            devices_[id].vram_free    = devices_[id].vram_total;
#endif
        }
    }

    // Remove job_id -> pid mapping
    for (auto jit = jobid_to_pid_.begin(); jit != jobid_to_pid_.end(); ) {
        if (jit->second == pid) jit = jobid_to_pid_.erase(jit);
        else ++jit;
    }

    pid_to_gpus_.erase(it);
}

void GPUStateTracker::refresh_vram() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& dev : devices_) {
#ifdef USE_MOCK_NVML
        if (!dev.is_busy) {
            dev.vram_free = dev.vram_total;
        }
        nvmlDeviceGetTemperature(dev.handle, NVML_TEMPERATURE_GPU, &dev.temperature);
#else
        nvmlMemory_t mem;
        nvmlReturn_t r = nvmlDeviceGetMemoryInfo(dev.handle, &mem);
        if (r == NVML_SUCCESS) {
            dev.vram_free = (size_t)mem.free;
        }
        // Also refresh temperature
        nvmlDeviceGetTemperature(dev.handle, NVML_TEMPERATURE_GPU, &dev.temperature);
#endif
    }
}

std::vector<GPUDevice> GPUStateTracker::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return devices_;
}

int GPUStateTracker::free_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int n = 0;
    for (const auto& d : devices_) if (!d.is_busy) n++;
    return n;
}

int GPUStateTracker::total_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (int)devices_.size();
}
