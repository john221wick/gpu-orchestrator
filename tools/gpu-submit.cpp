// gpu-submit: Submit a job to the gpu-scheduler daemon from a JSON file.
//
// Usage:
//   gpu-submit job.json
//   gpu-submit jobs/finetune.json
//   gpu-submit job.json --socket /tmp/gpu-scheduler.sock

#include "protocol.h"
#include "json.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <stdexcept>

using json = nlohmann::json;

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string path_dirname(const std::string& path) {
    std::filesystem::path fs_path(path);
    auto parent = fs_path.parent_path();
    if (parent.empty()) return ".";
    return parent.string();
}

static std::string resolve_relative_path(
    const std::string& maybe_relative,
    const std::string& base_dir)
{
    if (maybe_relative.empty()) return maybe_relative;

    std::filesystem::path path(maybe_relative);
    if (path.is_absolute()) return path.string();

    std::error_code ec;
    auto resolved = std::filesystem::weakly_canonical(
        std::filesystem::path(base_dir) / path, ec);
    if (!ec) return resolved.string();

    return (std::filesystem::path(base_dir) / path).lexically_normal().string();
}

static bool framework_uses_script_path(const std::string& framework) {
    return framework != "custom";
}

static std::string send_request(const std::string& socket_path, const std::string& msg) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error(std::string("socket: ") + strerror(errno));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        throw std::runtime_error(
            std::string("Cannot connect to ") + socket_path +
            ": " + strerror(errno) + "\nIs gpu-scheduler running?");
    }

    send(fd, msg.c_str(), msg.size(), 0);
    shutdown(fd, SHUT_WR);

    std::string response;
    char buf[4096];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof(buf) - 1, 0)) > 0) {
        buf[n] = '\0';
        response += buf;
    }
    close(fd);
    return response;
}

// Inject "type":"submit" into the JSON if it isn't already there
static std::string normalize_job_payload(
    const std::string& payload,
    const std::string& job_file)
{
    json request = json::parse(payload);
    std::string job_dir = resolve_relative_path(path_dirname(job_file), ".");

    if (request.contains("working_dir") && request["working_dir"].is_string()) {
        std::string working_dir = request["working_dir"].get<std::string>();
        if (!working_dir.empty()) {
            request["working_dir"] = resolve_relative_path(working_dir, job_dir);
        }
    }

    std::string framework = request.value("framework", std::string("python"));
    if (request.contains("script") &&
        request["script"].is_string() &&
        framework_uses_script_path(framework))
    {
        std::string script = request["script"].get<std::string>();
        std::string working_dir = request.value("working_dir", std::string());
        if (working_dir.empty()) {
            request["script"] = resolve_relative_path(script, job_dir);
        }
    }

    request["type"] = PROTO_TYPE_SUBMIT;
    return request.dump();
}

int main(int argc, char* argv[]) {
    std::string job_file;
    std::string socket_path = SCHEDULER_SOCKET_PATH;
    bool print_request = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (strcmp(argv[i], "--print-request") == 0) {
            print_request = true;
        } else if (argv[i][0] != '-') {
            job_file = argv[i];
        } else {
            fprintf(stderr, "Usage: %s <job.json> [--socket PATH] [--print-request]\n", argv[0]);
            return 1;
        }
    }

    if (job_file.empty()) {
        fprintf(stderr, "Usage: %s <job.json> [--socket PATH] [--print-request]\n\n", argv[0]);
        fprintf(stderr, "Example job.json:\n");
        fprintf(stderr, "  {\n");
        fprintf(stderr, "    \"framework\": \"deepspeed\",\n");
        fprintf(stderr, "    \"num_gpus\": 2,\n");
        fprintf(stderr, "    \"needs_peer\": true,\n");
        fprintf(stderr, "    \"script\": \"scripts/finetune_example.py\",\n");
        fprintf(stderr, "    \"args\": [\"--steps\", \"100\"]\n");
        fprintf(stderr, "  }\n");
        return 1;
    }

    try {
        std::string payload = read_file(job_file);
        std::string request = normalize_job_payload(payload, job_file);
        // Ensure newline terminator
        if (!request.empty() && request.back() != '\n') request += '\n';

        if (print_request) {
            fputs(request.c_str(), stdout);
            return 0;
        }

        std::string response = send_request(socket_path, request);
        json parsed = json::parse(response);

        std::string status = parsed.value("status", "");
        if (status == "accepted") {
            std::string job_id = parsed.value("job_id", "");
            printf("[OK] %s  (log: /tmp/gpu-job-%s.log)\n",
                   job_id.c_str(), job_id.c_str());
        } else {
            std::string msg = parsed.value("message", response);
            fprintf(stderr, "[ERROR] %s\n", msg.empty() ? response.c_str() : msg.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] %s\n", e.what());
        return 1;
    }

    return 0;
}
