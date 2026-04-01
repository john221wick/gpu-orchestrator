#include "process_monitor.h"
#include "logger.h"

#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <thread>
#include <chrono>

ProcessMonitor::ProcessMonitor(GPUStateTracker& state, OnJobComplete callback)
    : state_(state), on_complete_(std::move(callback))
{}

void ProcessMonitor::track(pid_t pid, const JobRequest& job) {
    std::lock_guard<std::mutex> lock(mutex_);
    TrackedJob tj;
    tj.job_id        = job.job_id;
    tj.pid           = pid;
    tj.started_at    = job.started_at ? job.started_at : time(nullptr);
    tj.max_time_sec  = job.max_time_sec;
    tj.gpu_ids       = job.assigned_gpus;
    tj.framework     = job.framework;
    tj.script        = job.script;
    tj.master_port   = job.master_port;
    tj.sigterm_sent  = false;
    tracked_[pid]    = std::move(tj);

    LOG_INFO("MONITOR", "Tracking PID=%d job=%s max_time=%ds",
             pid, job.job_id.c_str(), job.max_time_sec);
}

void ProcessMonitor::run() {
    LOG_INFO("MONITOR", "Reaper thread started");
    while (running_) {
        // Non-blocking waitpid for any child
        int status;
        pid_t pid = waitpid(-1, &status, WNOHANG);

        if (pid > 0) {
            int exit_code = 0;
            int master_port = 0;
            if (WIFEXITED(status)) {
                exit_code = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                exit_code = -(int)WTERMSIG(status);
            }

            std::string job_id;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = tracked_.find(pid);
                if (it != tracked_.end()) {
                    job_id = it->second.job_id;
                    master_port = it->second.master_port;
                    time_t started = it->second.started_at;
                    long runtime = (long)(time(nullptr) - started);
                    long mins = runtime / 60;
                    long secs = runtime % 60;
                    LOG_INFO("DONE", "Job %s (PID %d) exited code=%d (runtime: %ldm%lds)",
                             job_id.c_str(), pid, exit_code, mins, secs);
                    tracked_.erase(it);
                }
            }

            // Free GPUs and notify scheduler
            state_.mark_free(pid);
            if (!job_id.empty() && on_complete_) {
                on_complete_(job_id, pid, exit_code, master_port);
            }

        } else if (pid == 0) {
            // No children changed state — sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        } else {
            // pid < 0: error or no children
            if (errno == ECHILD) {
                // No children at all — sleep longer
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            } else {
                // Unexpected error
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        }
    }
    LOG_INFO("MONITOR", "Reaper thread stopped");
}

void ProcessMonitor::run_timeout_checker() {
    LOG_INFO("MONITOR", "Timeout checker thread started");
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        std::lock_guard<std::mutex> lock(mutex_);
        time_t now = time(nullptr);

        for (auto& [pid, tj] : tracked_) {
            if (tj.max_time_sec <= 0) continue;

            long elapsed = (long)(now - tj.started_at);
            if (elapsed >= tj.max_time_sec) {
                if (!tj.sigterm_sent) {
                    LOG_WARN("TIMEOUT", "Job %s (PID %d) exceeded %ds limit — sending SIGTERM",
                             tj.job_id.c_str(), pid, tj.max_time_sec);
                    kill(-pid, SIGTERM);  // kill process group
                    tj.sigterm_sent = true;
                } else if (elapsed >= tj.max_time_sec + 10) {
                    LOG_WARN("TIMEOUT", "Job %s (PID %d) still alive 10s after SIGTERM — sending SIGKILL",
                             tj.job_id.c_str(), pid);
                    kill(-pid, SIGKILL);
                }
            }
        }
    }
    LOG_INFO("MONITOR", "Timeout checker thread stopped");
}

void ProcessMonitor::stop() {
    running_ = false;
}

std::vector<ProcessMonitor::TrackedJob> ProcessMonitor::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<TrackedJob> result;
    for (const auto& [pid, tj] : tracked_) {
        result.push_back(tj);
    }
    return result;
}
