#include "gpu_device.h"
#include "topology.h"
#include "gpu_state.h"
#include "job_queue.h"
#include "scheduler.h"
#include "launcher.h"
#include "process_monitor.h"
#include "server.h"
#include "protocol.h"
#include "logger.h"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>
#include <atomic>
#include <stdexcept>

// Forward declaration
std::vector<GPUDevice> discover_gpus();

// Global flag for graceful shutdown
static std::atomic<bool> g_shutdown{false};

// Components that need cleanup on shutdown
static ProcessMonitor* g_monitor  = nullptr;
static Scheduler*      g_sched    = nullptr;
static Server*         g_server   = nullptr;

static void signal_handler(int sig) {
    (void)sig;
    const char* msg = "[SIGNAL] Shutdown requested\n";
    write(STDERR_FILENO, msg, strlen(msg));
    g_shutdown = true;
    if (g_server)  g_server->stop();
    if (g_sched)   g_sched->stop();
    if (g_monitor) g_monitor->stop();
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "  -s, --socket PATH    Unix socket path (default: " SCHEDULER_SOCKET_PATH ")\n"
        "  -l, --log FILE       Log to file in addition to stdout\n"
        "  -d, --debug          Enable debug logging\n"
        "  -h, --help           Show this help\n",
        prog);
}

int main(int argc, char* argv[]) {
    std::string socket_path = SCHEDULER_SOCKET_PATH;
    std::string log_file;
    bool debug = false;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--socket") == 0) && i + 1 < argc) {
            socket_path = argv[++i];
        } else if ((strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--log") == 0) && i + 1 < argc) {
            log_file = argv[++i];
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Initialize logger
    logger_init(log_file);
    if (debug) logger_set_level(LogLevel::DEBUG);

    LOG_INFO("INIT", "gpu-scheduler starting up");

    // 1. Discover GPUs via NVML
    std::vector<GPUDevice> devices;
    try {
        devices = discover_gpus();
    } catch (const std::exception& e) {
        LOG_ERROR("INIT", "GPU discovery failed: %s", e.what());
        return 1;
    }
    LOG_INFO("INIT", "Discovered %zu GPU(s)", devices.size());

    // 2. Build topology matrix
    TopologyMatrix topo;
    topo.build(devices);
    topo.print();

    // 3. Initialize state tracker
    GPUStateTracker state;
    state.init(devices);

    // 4. Create queues
    JobQueue pending_queue;
    JobQueue waiting_queue;
    Launcher* launcher_ptr = nullptr;

    // 5. Create ProcessMonitor
    ProcessMonitor monitor(state,
        [&](const std::string& job_id, pid_t pid, int exit_code, int master_port) {
            LOG_INFO("DONE", "Job %s (PID %d) finished with code %d",
                     job_id.c_str(), pid, exit_code);
            if (launcher_ptr && master_port > 0) {
                launcher_ptr->release_port(master_port);
            }
            // Scheduler will re-check waiting queue
            if (g_sched) g_sched->on_job_complete(job_id);
        });
    g_monitor = &monitor;

    // 6. Create Launcher
    Launcher launcher(state,
        [&](pid_t pid, const JobRequest& job) {
            monitor.track(pid, job);
        });
    launcher_ptr = &launcher;

    // 7. Create Scheduler
    Scheduler scheduler(pending_queue, waiting_queue, state, topo, launcher);
    g_sched = &scheduler;

    // 8. Create Server
    Server server(socket_path, pending_queue, state, monitor, waiting_queue,
                  [&scheduler]{ scheduler.notify_work_available(); });
    g_server = &server;

    // 9. Register signal handlers
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);  // Ignore broken pipe from closed sockets

    // 10. Launch all threads
    std::thread t_server  ([&]{ server.run(); });
    std::thread t_sched   ([&]{ scheduler.run(); });
    std::thread t_reaper  ([&]{ monitor.run(); });
    std::thread t_timeout ([&]{ monitor.run_timeout_checker(); });
    std::thread t_vram    ([&]{
        while (!g_shutdown) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            state.refresh_vram();
        }
    });

    LOG_INFO("READY", "gpu-scheduler listening on %s", socket_path.c_str());
    LOG_INFO("READY", "Submit jobs with: gpu-submit  |  Check status with: gpu-status");

    // Wait for server thread (it blocks on accept until stopped)
    t_server.join();

    // Signal other threads to stop
    g_shutdown = true;
    scheduler.stop();
    monitor.stop();
    pending_queue.notify_all();
    waiting_queue.notify_all();

    LOG_INFO("SHUTDOWN", "Waiting for threads to finish...");
    t_sched.join();
    t_reaper.join();
    t_timeout.join();
    t_vram.join();

    LOG_INFO("SHUTDOWN", "gpu-scheduler exited cleanly");
    return 0;
}
