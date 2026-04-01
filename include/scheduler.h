#pragma once
#include "job_queue.h"
#include "gpu_state.h"
#include "topology.h"
#include "launcher.h"
#include <mutex>
#include <condition_variable>
#include <atomic>

class Scheduler {
public:
    Scheduler(JobQueue& pending, JobQueue& waiting,
              GPUStateTracker& state, TopologyMatrix& topo,
              Launcher& launcher);

    // Main scheduling loop (runs in its own thread).
    // Pops from pending queue, tries to assign GPUs,
    // if no GPUs available, moves to waiting queue.
    void run();

    // Called by reaper when a job finishes — re-checks waiting queue
    void on_job_complete(const std::string& job_id);

    // Called when new work lands in the pending queue.
    void notify_work_available();

    void stop();

private:
    bool try_schedule(JobRequest& job);
    void reschedule_waiting();

    JobQueue&         pending_;
    JobQueue&         waiting_;
    GPUStateTracker&  state_;
    TopologyMatrix&   topo_;
    Launcher&         launcher_;

    std::mutex              reschedule_mutex_;
    std::condition_variable reschedule_cv_;
    std::atomic<bool>       work_available_{true};
    std::atomic<bool>       running_{true};
};
