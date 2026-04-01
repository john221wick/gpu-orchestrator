// gpu-status: CLI client to display gpu-scheduler daemon status
//
// Usage:
//   gpu-status [--socket PATH] [--json]

#include "protocol.h"
#include "json.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

static std::string send_request(const std::string& socket_path, const std::string& msg) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error(std::string("socket: ") + strerror(errno));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        throw std::runtime_error(std::string("connect(") + socket_path + "): " + strerror(errno)
            + "\nIs gpu-scheduler running?");
    }

    send(fd, msg.c_str(), msg.size(), 0);
    shutdown(fd, SHUT_WR);

    std::string response;
    char buf[65536];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof(buf) - 1, 0)) > 0) {
        buf[n] = '\0';
        response += buf;
    }
    close(fd);
    return response;
}

struct GPUInfo {
    int id = 0;
    std::string name;
    double vram_total_gb = 0;
    double vram_free_gb  = 0;
    int temperature = 0;
    bool busy = false;
    std::string job_id;
};

struct JobInfo {
    std::string job_id;
    std::string framework;
    std::string script;
    std::vector<int> gpus;
    long runtime_sec = 0;
};

static std::vector<GPUInfo> parse_gpus(const json& parsed) {
    std::vector<GPUInfo> result;
    if (!parsed.contains("gpus") || !parsed["gpus"].is_array()) return result;
    for (const auto& item : parsed["gpus"]) {
        GPUInfo g;
        g.id = item.value("id", 0);
        g.name = item.value("name", std::string());
        g.vram_total_gb = item.value("vram_total_gb", 0.0);
        g.vram_free_gb = item.value("vram_free_gb", 0.0);
        g.temperature = item.value("temperature", 0);
        g.busy = item.value("busy", false);
        if (item.contains("job_id") && !item["job_id"].is_null()) {
            g.job_id = item["job_id"].get<std::string>();
        }
        result.push_back(g);
    }
    return result;
}

static std::vector<JobInfo> parse_running_jobs(const json& parsed) {
    std::vector<JobInfo> result;
    if (!parsed.contains("jobs_running") || !parsed["jobs_running"].is_array()) return result;
    for (const auto& item : parsed["jobs_running"]) {
        JobInfo j;
        j.job_id = item.value("job_id", std::string());
        j.framework = item.value("framework", std::string());
        j.script = item.value("script", std::string());
        j.runtime_sec = item.value("runtime_sec", 0L);
        if (item.contains("gpus")) {
            j.gpus = item["gpus"].get<std::vector<int>>();
        }
        result.push_back(j);
    }
    return result;
}

static int parse_queued_count(const json& parsed) {
    return parsed.value("jobs_queued", 0);
}

// ---------------------------------------------------------------------------
// Pretty-print
// ---------------------------------------------------------------------------
static void print_bar(int width) {
    printf("\u2550");
    for (int i = 0; i < width - 2; i++) printf("\u2550");
    printf("\u2550\n");
}

static std::string runtime_str(long secs) {
    if (secs < 60) return std::to_string(secs) + "s";
    long m = secs / 60, s = secs % 60;
    if (m < 60) {
        char buf[32]; snprintf(buf, sizeof(buf), "%ldm%02lds", m, s);
        return buf;
    }
    long h = m / 60; m %= 60;
    char buf[32]; snprintf(buf, sizeof(buf), "%ldh%02ldm", h, m);
    return buf;
}

static void print_status(const std::string& response_json, bool raw_json) {
    if (raw_json) { puts(response_json.c_str()); return; }

    json parsed = json::parse(response_json);
    auto gpus    = parse_gpus(parsed);
    auto jobs    = parse_running_jobs(parsed);
    int  queued  = parse_queued_count(parsed);

    int total_gpus = (int)gpus.size();
    int free_gpus  = 0;
    for (auto& g : gpus) if (!g.busy) free_gpus++;

    const int W = 66;
    printf("\u2554"); print_bar(W);
    printf("\u2551  %-*s\u2551\n", W - 2, "             GPU SCHEDULER STATUS");
    printf("\u2560"); print_bar(W);

    for (const auto& g : gpus) {
        std::string state_str = g.busy ? "BUSY" : "FREE";
        std::string vram_str;
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "VRAM: %4.1f/%4.1f GB",
                     g.vram_total_gb - g.vram_free_gb, g.vram_total_gb);
            vram_str = buf;
        }
        std::string temp_str;
        if (g.temperature > 0) {
            char buf[16]; snprintf(buf, sizeof(buf), "%u°C", g.temperature);
            temp_str = buf;
        }

        // Truncate GPU name to 18 chars
        std::string name = g.name;
        if (name.size() > 18) name = name.substr(0, 15) + "...";

        std::string line;
        {
            char buf[128];
            if (g.busy) {
                snprintf(buf, sizeof(buf), " GPU %-2d: %-4s  %-18s  %-20s  %s",
                         g.id, state_str.c_str(), name.c_str(),
                         vram_str.c_str(), g.job_id.c_str());
            } else {
                snprintf(buf, sizeof(buf), " GPU %-2d: %-4s  %-18s  %s",
                         g.id, state_str.c_str(), name.c_str(), vram_str.c_str());
            }
            line = buf;
        }
        printf("\u2551 %-*s \u2551\n", W - 3, line.c_str());
    }

    if (!jobs.empty()) {
        printf("\u2560"); print_bar(W);
        printf("\u2551  %-*s\u2551\n", W - 2, "Running Jobs:");
        for (const auto& j : jobs) {
            std::string gpu_str;
            for (size_t i = 0; i < j.gpus.size(); i++) {
                if (i) gpu_str += ',';
                gpu_str += std::to_string(j.gpus[i]);
            }
            char buf[128];
            snprintf(buf, sizeof(buf), " [%s] %-10s  GPU %-6s  %s  %s",
                     j.job_id.c_str(), j.framework.c_str(), gpu_str.c_str(),
                     runtime_str(j.runtime_sec).c_str(), j.script.c_str());
            printf("\u2551 %-*s \u2551\n", W - 3, buf);
        }
    }

    printf("\u2560"); print_bar(W);
    {
        char summary[128];
        snprintf(summary, sizeof(summary),
                 " Running: %d job%s  |  Queued: %d  |  Free GPUs: %d/%d",
                 (int)jobs.size(), jobs.size() == 1 ? "" : "s",
                 queued, free_gpus, total_gpus);
        printf("\u2551 %-*s \u2551\n", W - 3, summary);
    }
    printf("\u255a"); print_bar(W);
}

int main(int argc, char* argv[]) {
    std::string socket_path = SCHEDULER_SOCKET_PATH;
    bool raw_json = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0) {
            raw_json = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: %s [--socket PATH] [--json]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    std::string request = "{\"type\":\"status\"}\n";
    try {
        std::string response = send_request(socket_path, request);
        print_status(response, raw_json);
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] %s\n", e.what());
        return 1;
    }

    return 0;
}
