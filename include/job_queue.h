#pragma once
#include "job.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <vector>

// Jobs ordered by: (1) priority tier (lower = higher priority),
// then (2) submit time (earlier = higher priority)
struct JobCompare {
    bool operator()(const JobRequest& a, const JobRequest& b) const {
        if (a.priority != b.priority)
            return a.priority > b.priority;
        return a.submitted_at > b.submitted_at;
    }
};

class JobQueue {
public:
    void push(JobRequest job);

    // Blocks until a job is available
    JobRequest pop();

    // Non-blocking: returns nullopt if empty
    std::optional<JobRequest> try_pop();

    size_t size() const;
    bool empty() const;

    // Drain all jobs (for status display)
    std::vector<JobRequest> snapshot() const;

    // Notify waiting threads to wake up (used on stop)
    void notify_all();

private:
    std::priority_queue<JobRequest, std::vector<JobRequest>, JobCompare> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};
