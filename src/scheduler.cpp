#include "scheduler.h"
#include "logger.h"
#include <thread>
#include <chrono>
#include <sstream>

Scheduler::Scheduler(JobQueue& pending, JobQueue& waiting,
                     GPUStateTracker& state, TopologyMatrix& topo,
                     Launcher& launcher)
    : pending_(pending), waiting_(waiting),
      state_(state), topo_(topo), launcher_(launcher)
{}

bool Scheduler::try_schedule(JobRequest& job) {
    // Find available GPUs meeting the job's requirements
    std::vector<int> candidates = state_.find_available(
        job.num_gpus, job.min_vram, job.needs_peer, topo_);

    if (candidates.empty()) {
        LOG_INFO("QUEUED", "Job %s: not enough resources (%d GPUs, %.1f GB VRAM each) — waiting",
                 job.job_id.c_str(), job.num_gpus,
                 (double)job.min_vram / (1024.0 * 1024.0 * 1024.0));
        job.status = JobRequest::QUEUED;
        waiting_.push(job);
        return false;
    }

    // Log topology scoring if multi-GPU with peer requirement
    if (job.needs_peer && job.num_gpus > 1) {
        LOG_INFO("SCHED", "Evaluating GPU groups for %d-GPU peer job:", job.num_gpus);

        // Log scores for all candidate combinations (up to first few)
        int n = (int)candidates.size();
        if (n <= 4) {
            // Show all pairs/groups for small sets
            std::function<void(int, std::vector<int>&)> show =
                [&](int start, std::vector<int>& cur) {
                    if ((int)cur.size() == job.num_gpus) {
                        int score = topo_.score_group(cur);
                        std::string ids;
                        for (int i = 0; i < (int)cur.size(); i++) {
                            if (i) ids += ',';
                            ids += std::to_string(cur[i]);
                        }
                        LOG_INFO("SCHED", "  {%s} score=%d", ids.c_str(), score);
                        return;
                    }
                    for (int i = start; i <= n - (job.num_gpus - (int)cur.size()); i++) {
                        cur.push_back(candidates[i]);
                        show(i + 1, cur);
                        cur.pop_back();
                    }
                };
            std::vector<int> cur;
            show(0, cur);
        }

        int best_score = topo_.score_group(candidates);
        LOG_INFO("SCHED", "  BEST: score=%d", best_score);
    }

    // Assign GPUs to job
    job.assigned_gpus = candidates;

    // Log assignment
    std::string gpu_str;
    for (size_t i = 0; i < candidates.size(); i++) {
        if (i) gpu_str += ',';
        gpu_str += std::to_string(candidates[i]);
    }
    LOG_INFO("ASSIGN", "Job %s -> GPU %s (framework=%s, user=%s)",
             job.job_id.c_str(), gpu_str.c_str(),
             framework_to_str(job.framework), job.user_id.c_str());

    try {
        launcher_.launch(job);
    } catch (const std::exception& e) {
        job.status = JobRequest::FAILED;
        LOG_ERROR("LAUNCH", "Job %s failed to launch: %s",
                  job.job_id.c_str(), e.what());
        return false;
    }

    return true;
}

void Scheduler::reschedule_waiting() {
    // Drain the waiting queue and re-attempt scheduling
    // We re-push jobs that still can't be scheduled
    std::vector<JobRequest> still_waiting;

    while (true) {
        auto opt = waiting_.try_pop();
        if (!opt) break;
        JobRequest job = std::move(*opt);

        std::vector<int> candidates = state_.find_available(
            job.num_gpus, job.min_vram, job.needs_peer, topo_);

        if (!candidates.empty()) {
            job.assigned_gpus = candidates;
            std::string gpu_str;
            for (size_t i = 0; i < candidates.size(); i++) {
                if (i) gpu_str += ',';
                gpu_str += std::to_string(candidates[i]);
            }
            LOG_INFO("ASSIGN", "[RE-SCHED] Job %s -> GPU %s",
                     job.job_id.c_str(), gpu_str.c_str());
            try {
                launcher_.launch(job);
            } catch (const std::exception& e) {
                job.status = JobRequest::FAILED;
                LOG_ERROR("LAUNCH", "Waiting job %s failed to launch: %s",
                          job.job_id.c_str(), e.what());
            }
        } else {
            still_waiting.push_back(std::move(job));
        }
    }

    // Re-push jobs that still couldn't be scheduled
    for (auto& j : still_waiting) {
        waiting_.push(std::move(j));
    }
}

void Scheduler::run() {
    LOG_INFO("SCHED", "Scheduler thread started");
    while (running_) {
        {
            std::unique_lock<std::mutex> lock(reschedule_mutex_);
            reschedule_cv_.wait(lock, [this]{
                return !running_.load()
                    || work_available_.load()
                    || !pending_.empty()
                    || !waiting_.empty();
            });
            work_available_ = false;
        }

        if (!running_) break;

        // Try to schedule any waiting jobs first (higher priority — they've been waiting)
        if (!waiting_.empty()) {
            reschedule_waiting();
        }

        // Then process new pending jobs
        while (true) {
            auto opt = pending_.try_pop();
            if (!opt) break;

            JobRequest job = std::move(*opt);
            LOG_INFO("SCHED", "Processing job %s (framework=%s, gpus=%d, peer=%s)",
                     job.job_id.c_str(), framework_to_str(job.framework),
                     job.num_gpus, job.needs_peer ? "yes" : "no");

            try_schedule(job);
        }
    }
    LOG_INFO("SCHED", "Scheduler thread stopped");
}

void Scheduler::on_job_complete(const std::string& job_id) {
    (void)job_id;
    notify_work_available();
}

void Scheduler::notify_work_available() {
    {
        std::lock_guard<std::mutex> lock(reschedule_mutex_);
        work_available_ = true;
    }
    reschedule_cv_.notify_one();
}

void Scheduler::stop() {
    running_ = false;
    reschedule_cv_.notify_all();
}
