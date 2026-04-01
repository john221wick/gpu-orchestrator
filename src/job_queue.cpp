#include "job_queue.h"

void JobQueue::push(JobRequest job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(job));
    }
    cv_.notify_one();
}

JobRequest JobQueue::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]{ return !queue_.empty(); });
    JobRequest job = queue_.top();
    queue_.pop();
    return job;
}

std::optional<JobRequest> JobQueue::try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    JobRequest job = queue_.top();
    queue_.pop();
    return job;
}

size_t JobQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool JobQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

std::vector<JobRequest> JobQueue::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    // Copy the priority queue by dumping to a vector
    auto copy = queue_;
    std::vector<JobRequest> result;
    while (!copy.empty()) {
        result.push_back(copy.top());
        copy.pop();
    }
    return result;
}

void JobQueue::notify_all() {
    cv_.notify_all();
}
