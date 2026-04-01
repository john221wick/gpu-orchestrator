#pragma once
#include "job_queue.h"
#include "gpu_state.h"
#include "process_monitor.h"
#include <string>
#include <atomic>
#include <functional>

class Server {
public:
    Server(const std::string& socket_path, JobQueue& queue,
           GPUStateTracker& state, ProcessMonitor& monitor, JobQueue& waiting,
           std::function<void()> on_work_available = {});
    ~Server();

    // Main listen loop (runs in its own thread)
    void run();

    void stop();

private:
    // Handle a single client connection (called per accept)
    void handle_client(int client_fd);

    // Parse JSON submit request into JobRequest
    JobRequest parse_submit(const std::string& payload);

    // Build JSON status response
    std::string build_status_response();

    std::string      socket_path_;
    JobQueue&        pending_;
    GPUStateTracker& state_;
    ProcessMonitor&  monitor_;
    JobQueue&        waiting_;
    std::function<void()> on_work_available_;
    int              server_fd_ = -1;
    std::atomic<bool> running_{true};
};
