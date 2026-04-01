#include "logger.h"
#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <mutex>
#include <string>

static FILE*    g_log_file = nullptr;
static LogLevel g_min_level = LogLevel::INFO;
static std::mutex g_log_mutex;

void logger_init(const std::string& log_file) {
    if (!log_file.empty()) {
        g_log_file = fopen(log_file.c_str(), "a");
        if (!g_log_file) {
            fprintf(stderr, "[WARN] Could not open log file: %s\n", log_file.c_str());
        }
    }
}

void logger_set_level(LogLevel level) {
    g_min_level = level;
}

void log_msg(LogLevel level, const char* tag, const char* fmt, ...) {
    if (level < g_min_level) return;

    // Timestamp
    time_t now = time(nullptr);
    struct tm tm_info;
    localtime_r(&now, &tm_info);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &tm_info);

    const char* level_str = "INFO";
    switch (level) {
        case LogLevel::DEBUG: level_str = "DEBUG"; break;
        case LogLevel::INFO:  level_str = "INFO";  break;
        case LogLevel::WARN:  level_str = "WARN";  break;
        case LogLevel::ERROR: level_str = "ERROR"; break;
    }

    // Format message
    char msg_buf[4096];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
    va_end(args);

    std::lock_guard<std::mutex> lock(g_log_mutex);
    fprintf(stdout, "[%s] [%s] %s\n", timestamp, tag, msg_buf);
    fflush(stdout);

    if (g_log_file) {
        fprintf(g_log_file, "[%s] [%s] [%s] %s\n", timestamp, level_str, tag, msg_buf);
        fflush(g_log_file);
    }
}
