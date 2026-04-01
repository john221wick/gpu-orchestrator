#pragma once
#include <cstdio>
#include <string>

// Log levels
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

void logger_init(const std::string& log_file = "");
void logger_set_level(LogLevel level);
void log_msg(LogLevel level, const char* tag, const char* fmt, ...);

// Convenience macros
#define LOG_INFO(tag, ...)  log_msg(LogLevel::INFO,  tag, __VA_ARGS__)
#define LOG_WARN(tag, ...)  log_msg(LogLevel::WARN,  tag, __VA_ARGS__)
#define LOG_ERROR(tag, ...) log_msg(LogLevel::ERROR, tag, __VA_ARGS__)
#define LOG_DEBUG(tag, ...) log_msg(LogLevel::DEBUG, tag, __VA_ARGS__)
