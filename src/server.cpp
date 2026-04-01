#include "server.h"
#include "protocol.h"
#include "logger.h"
#include "json.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <stdexcept>
#include <thread>

using json = nlohmann::json;

// Simple UUID generator
static std::string generate_job_id() {
    unsigned char buf[4];
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        if (read(fd, buf, 4) != 4) {
            // fallback
            buf[0] = (unsigned char)(rand() & 0xFF);
            buf[1] = (unsigned char)(rand() & 0xFF);
            buf[2] = (unsigned char)(rand() & 0xFF);
            buf[3] = (unsigned char)(rand() & 0xFF);
        }
        close(fd);
    } else {
        buf[0] = buf[1] = buf[2] = buf[3] = (unsigned char)(rand() & 0xFF);
    }
    char id[16];
    snprintf(id, sizeof(id), "job-%02x%02x%02x%02x", buf[0], buf[1], buf[2], buf[3]);
    return id;
}

// ---------------------------------------------------------------------------

Server::Server(const std::string& socket_path, JobQueue& queue,
               GPUStateTracker& state, ProcessMonitor& monitor, JobQueue& waiting,
               std::function<void()> on_work_available)
    : socket_path_(socket_path), pending_(queue),
      state_(state), monitor_(monitor), waiting_(waiting),
      on_work_available_(std::move(on_work_available))
{}

Server::~Server() {
    if (server_fd_ >= 0) {
        close(server_fd_);
        unlink(socket_path_.c_str());
    }
}

void Server::run() {
    // Create Unix domain socket
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        LOG_ERROR("SERVER", "socket() failed: %s", strerror(errno));
        return;
    }

    // Allow reuse
    unlink(socket_path_.c_str());

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR("SERVER", "bind(%s) failed: %s", socket_path_.c_str(), strerror(errno));
        return;
    }

    if (listen(server_fd_, 16) < 0) {
        LOG_ERROR("SERVER", "listen() failed: %s", strerror(errno));
        return;
    }

    LOG_INFO("READY", "Listening on %s", socket_path_.c_str());

    while (running_) {
        struct sockaddr_un client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (running_) {
                LOG_WARN("SERVER", "accept() failed: %s", strerror(errno));
            }
            continue;
        }

        // Handle each client in a detached thread
        std::thread([this, client_fd]{ handle_client(client_fd); }).detach();
    }
}

void Server::handle_client(int client_fd) {
    // Read until newline or EOF
    std::string buf;
    buf.reserve(4096);
    char tmp[512];
    while (true) {
        ssize_t n = recv(client_fd, tmp, sizeof(tmp) - 1, 0);
        if (n <= 0) break;
        tmp[n] = '\0';
        buf += tmp;
        if (buf.find('\n') != std::string::npos) break;
        if (buf.size() > PROTO_MAX_MSG_SIZE) break;
    }

    if (buf.empty()) {
        close(client_fd);
        return;
    }

    std::string response;

    try {
        json request = json::parse(buf);
        std::string type = request.value("type", "");

        if (type == PROTO_TYPE_SUBMIT) {
            JobRequest job = parse_submit(request.dump());
            LOG_INFO("RECV", "Job %s from user '%s': %s, %d GPU(s)",
                     job.job_id.c_str(), job.user_id.c_str(),
                     framework_to_str(job.framework), job.num_gpus);
            pending_.push(job);
            if (on_work_available_) on_work_available_();
            response = "{\"status\":\"" PROTO_STATUS_ACCEPTED "\","
                       "\"job_id\":\"" + job.job_id + "\"}\n";
        } else if (type == PROTO_TYPE_STATUS) {
            response = build_status_response();
        } else {
            response = "{\"status\":\"error\",\"message\":\"unknown type\"}\n";
        }
    } catch (const std::exception& e) {
        LOG_ERROR("SERVER", "Failed to handle request: %s", e.what());
        response = "{\"status\":\"" PROTO_STATUS_ERROR "\","
                   "\"message\":\"" + std::string(e.what()) + "\"}\n";
    }

    send(client_fd, response.c_str(), response.size(), 0);
    close(client_fd);
}

JobRequest Server::parse_submit(const std::string& payload) {
    json request = json::parse(payload);
    JobRequest job;
    job.job_id       = generate_job_id();
    job.submitted_at = time(nullptr);

    if (!request.is_object()) {
        throw std::runtime_error("request must be a JSON object");
    }

    if (request.contains("user_id") && request["user_id"].is_string()) {
        job.user_id = request["user_id"].get<std::string>();
    } else {
        job.user_id = request.value("user", std::string("anonymous"));
    }

    job.framework    = str_to_framework(request.value("framework", std::string("python")));
    job.job_type     = str_to_jobtype(request.value("job_type", std::string("inference")));
    job.num_gpus     = request.value("num_gpus", 1);
    job.needs_peer   = request.value("needs_peer", false);
    job.priority     = request.value("priority", 5);
    job.script       = request.value("script", std::string());
    job.working_dir  = request.value("working_dir", std::string());
    job.max_time_sec = request.value("max_time_sec", 0);

    // Ray-specific
    std::string ray_addr = request.value("ray_address", std::string());
    if (!ray_addr.empty()) job.ray_address = ray_addr;

    if (request.contains("args")) {
        if (!request["args"].is_array()) {
            throw std::runtime_error("'args' must be an array of strings");
        }
        job.args = request["args"].get<std::vector<std::string>>();
    }

    if (request.contains("min_vram_gb")) {
        double vram_gb = request["min_vram_gb"].get<double>();
        job.min_vram = (size_t)(vram_gb * 1024.0 * 1024.0 * 1024.0);
    } else if (request.contains("min_vram")) {
        job.min_vram = request["min_vram"].get<size_t>();
    }

    if (job.script.empty()) {
        throw std::runtime_error("missing 'script' field");
    }
    if (job.num_gpus < 1 || job.num_gpus > 16) {
        throw std::runtime_error("num_gpus must be 1-16");
    }

    return job;
}

std::string Server::build_status_response() {
    json response;

    // GPU list
    auto gpus = state_.snapshot();
    response["gpus"] = json::array();
    for (const auto& g : gpus) {
        double total_gb = (double)g.vram_total / (1024.0 * 1024.0 * 1024.0);
        double free_gb  = (double)g.vram_free  / (1024.0 * 1024.0 * 1024.0);
        json gpu = {
            {"id", g.id},
            {"name", g.name},
            {"vram_total_gb", total_gb},
            {"vram_free_gb", free_gb},
            {"temperature", g.temperature},
            {"busy", g.is_busy}
        };
        if (g.is_busy) {
            gpu["job_id"] = g.owner_job_id;
        } else {
            gpu["job_id"] = nullptr;
        }
        response["gpus"].push_back(std::move(gpu));
    }

    // Running jobs
    auto running = monitor_.snapshot();
    response["jobs_running"] = json::array();
    for (const auto& j : running) {
        long runtime = (long)(time(nullptr) - j.started_at);
        response["jobs_running"].push_back({
            {"job_id", j.job_id},
            {"framework", framework_to_str(j.framework)},
            {"script", j.script},
            {"gpus", j.gpu_ids},
            {"runtime_sec", runtime}
        });
    }

    // Queued job count:
    // include both jobs already classified as waiting and jobs still sitting
    // in the pending queue awaiting a scheduler pass.
    size_t pending_count = pending_.size();
    size_t waiting_count = waiting_.size();
    response["jobs_pending"] = pending_count;
    response["jobs_waiting"] = waiting_count;
    response["jobs_queued"] = pending_count + waiting_count;
    return response.dump(2) + "\n";
}

void Server::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
        unlink(socket_path_.c_str());
    }
}
