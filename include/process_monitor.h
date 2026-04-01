#pragma once
#include "job.h"
#include "gpu_state.h"
#include <unordered_map>
#include <mutex>
#include <functional>
#include <atomic>

class ProcessMonitor {
public:
    using OnJobComplete = std::function<void(const std::string& job_id, pid_t pid,
                                            int exit_code, int master_port)>;

    ProcessMonitor(GPUStateTracker& state, OnJobComplete callback);

    // Register a launched job for monitoring
    void track(pid_t pid, const JobRequest& job);

    // Main reaper loop (runs in its own thread).
    // Calls waitpid(), frees GPUs, invokes callback.
    void run();

    // Timeout enforcer loop (runs in its own thread).
    // Checks if any job exceeded max_time_sec, sends SIGTERM then SIGKILL.
    void run_timeout_checker();

    void stop();

    // Get snapshot of running jobs for status display
    struct TrackedJob {
        std::string         job_id;
        pid_t               pid;
        time_t              started_at;
        int                 max_time_sec;
        std::vector<int>    gpu_ids;
        Framework           framework;
        std::string         script;
        int                 master_port = 0;
        bool                sigterm_sent = false;
    };
    std::vector<TrackedJob> snapshot() const;

private:
    GPUStateTracker&                            state_;
    OnJobComplete                               on_complete_;
    std::unordered_map<pid_t, TrackedJob>       tracked_;
    mutable std::mutex                          mutex_;
    std::atomic<bool>                           running_{true};
};
