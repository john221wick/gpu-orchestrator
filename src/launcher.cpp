#include "launcher.h"
#include "logger.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <sstream>
#include <stdexcept>

namespace {

bool command_exists_in_path(const char* command) {
    if (command == nullptr || *command == '\0') return false;
    if (strchr(command, '/')) return access(command, X_OK) == 0;

    const char* path = getenv("PATH");
    if (path == nullptr) return false;

    std::string paths(path);
    size_t start = 0;
    while (start <= paths.size()) {
        size_t end = paths.find(':', start);
        std::string dir = (end == std::string::npos)
            ? paths.substr(start)
            : paths.substr(start, end - start);
        if (dir.empty()) dir = ".";

        std::string candidate = dir + "/" + command;
        if (access(candidate.c_str(), X_OK) == 0) return true;

        if (end == std::string::npos) break;
        start = end + 1;
    }
    return false;
}

std::string select_python_command() {
    if (command_exists_in_path("python"))  return "python";
    if (command_exists_in_path("python3")) return "python3";
    return "python3";
}

std::string make_absolute_path(const std::string& path) {
    if (path.empty() || path[0] == '/') return path;

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) return path;
    return std::string(cwd) + "/" + path;
}

std::string select_project_command(const JobRequest& job, const char* command) {
    if (!job.working_dir.empty()) {
        std::string candidate = make_absolute_path(job.working_dir) + "/.venv/bin/" + command;
        if (access(candidate.c_str(), X_OK) == 0) return candidate;
    }

    if (command_exists_in_path(command)) return command;
    return std::string(command);
}

std::string select_python_command(const JobRequest& job) {
    if (!job.working_dir.empty()) {
        std::string candidate = make_absolute_path(job.working_dir) + "/.venv/bin/python";
        if (access(candidate.c_str(), X_OK) == 0) return candidate;
    }
    return select_python_command();
}

std::string select_accelerate_mixed_precision(const JobRequest& job) {
    std::string mixed_precision = "bf16";

    for (const auto& arg : job.args) {
        if (arg == "--bf16") {
            mixed_precision = "bf16";
        } else if (arg == "--fp16") {
            mixed_precision = "fp16";
        } else if (arg == "--no-bf16" && mixed_precision == "bf16") {
            mixed_precision = "no";
        } else if (arg == "--no-fp16" && mixed_precision == "fp16") {
            mixed_precision = "no";
        }
    }

    return mixed_precision;
}

}  // namespace

Launcher::Launcher(GPUStateTracker& state, OnLaunch on_launch)
    : state_(state), on_launch_(on_launch)
{}

std::string Launcher::build_gpu_list(const std::vector<int>& gpus) {
    std::string result;
    for (size_t i = 0; i < gpus.size(); i++) {
        if (i > 0) result += ',';
        result += std::to_string(gpus[i]);
    }
    return result;
}

int Launcher::allocate_master_port() {
    std::lock_guard<std::mutex> lock(port_mutex_);
    // Find a port not currently in use
    while (used_ports_.count(next_port_)) {
        next_port_++;
        if (next_port_ > 30000) next_port_ = 29500;
    }
    int port = next_port_++;
    used_ports_.insert(port);
    return port;
}

void Launcher::release_port(int port) {
    std::lock_guard<std::mutex> lock(port_mutex_);
    used_ports_.erase(port);
}

pid_t Launcher::do_fork_exec(
    const JobRequest& job,
    const std::vector<std::string>& argv,
    const std::vector<std::pair<std::string,std::string>>& env_extras)
{
    // Build argv as char* array
    std::vector<const char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto& a : argv) cargv.push_back(a.c_str());
    cargv.push_back(nullptr);

    // Log the command
    std::string cmd_str;
    for (const auto& a : argv) { cmd_str += a; cmd_str += ' '; }
    LOG_INFO("LAUNCH", "Command: %s", cmd_str.c_str());

    pid_t pid = fork();
    if (pid < 0) {
        LOG_ERROR("LAUNCH", "fork() failed: %s", strerror(errno));
        throw std::runtime_error("fork failed");
    }

    if (pid == 0) {
        // ---- CHILD PROCESS ----

        // IMPORTANT: Do NOT call NVML after fork — only exec'd process is clean.

        // Change working directory
        if (!job.working_dir.empty()) {
            if (chdir(job.working_dir.c_str()) != 0) {
                fprintf(stderr, "[LAUNCH] chdir to %s failed: %s\n",
                        job.working_dir.c_str(), strerror(errno));
            }
        }

        // Set extra environment variables
        for (const auto& kv : env_extras) {
            setenv(kv.first.c_str(), kv.second.c_str(), 1);
        }

        // Redirect stdout/stderr to a log file per job
        std::string log_path = "/tmp/gpu-job-" + job.job_id + ".log";
        int log_fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (log_fd >= 0) {
            dup2(log_fd, STDOUT_FILENO);
            dup2(log_fd, STDERR_FILENO);
            close(log_fd);
        }

        // Create new process group so we can kill the whole group on timeout
        setsid();

        execvp(cargv[0], const_cast<char* const*>(cargv.data()));

        // exec failed
        fprintf(stderr, "[LAUNCH] execvp(%s) failed: %s\n", cargv[0], strerror(errno));
        _exit(127);
    }

    // ---- PARENT PROCESS ----
    return pid;
}

void Launcher::launch(JobRequest& job) {
    switch (job.framework) {
        case Framework::TORCHRUN:    launch_torchrun(job);    break;
        case Framework::DEEPSPEED:   launch_deepspeed(job);   break;
        case Framework::ACCELERATE:  launch_accelerate(job);  break;
        case Framework::PYTHON:      launch_python(job);      break;
        case Framework::CUSTOM:      launch_custom(job);      break;
        case Framework::HF_TRAINER:  launch_hf_trainer(job);  break;
        case Framework::RAY_TRAIN:   launch_ray_train(job);   break;
        case Framework::RAY_JOB:     launch_ray_job(job);     break;
    }
}

void Launcher::launch_torchrun(JobRequest& job) {
    std::string gpu_list = build_gpu_list(job.assigned_gpus);
    int port = allocate_master_port();
    std::string torchrun_cmd = select_project_command(job, "torchrun");

    std::vector<std::string> argv = {
        torchrun_cmd,
        "--nproc_per_node=" + std::to_string(job.num_gpus),
        "--master_addr=127.0.0.1",
        "--master_port=" + std::to_string(port),
        job.script
    };
    for (const auto& arg : job.args) argv.push_back(arg);

    std::vector<std::pair<std::string,std::string>> env = {
        {"CUDA_VISIBLE_DEVICES", gpu_list}
    };

    pid_t pid = do_fork_exec(job, argv, env);
    state_.mark_busy(job.assigned_gpus, pid, job.job_id, job.min_vram);
    job.pids.push_back(pid);
    job.started_at = time(nullptr);
    job.master_port = port;
    job.status = JobRequest::RUNNING;

    LOG_INFO("LAUNCH", "torchrun PID=%d  CUDA_VISIBLE_DEVICES=%s  port=%d  log=/tmp/gpu-job-%s.log",
             pid, gpu_list.c_str(), port, job.job_id.c_str());

    if (on_launch_) on_launch_(pid, job);
}

void Launcher::launch_deepspeed(JobRequest& job) {
    std::string gpu_list = build_gpu_list(job.assigned_gpus);
    int port = allocate_master_port();
    std::string deepspeed_cmd = select_project_command(job, "deepspeed");

    std::vector<std::string> argv = {
        deepspeed_cmd,
        "--num_gpus=" + std::to_string(job.num_gpus),
        "--master_port=" + std::to_string(port),
        job.script
    };
    for (const auto& arg : job.args) argv.push_back(arg);

    std::vector<std::pair<std::string,std::string>> env = {
        {"CUDA_VISIBLE_DEVICES", gpu_list}
    };

    pid_t pid = do_fork_exec(job, argv, env);
    state_.mark_busy(job.assigned_gpus, pid, job.job_id, job.min_vram);
    job.pids.push_back(pid);
    job.started_at = time(nullptr);
    job.master_port = port;
    job.status = JobRequest::RUNNING;

    LOG_INFO("LAUNCH", "deepspeed PID=%d  CUDA_VISIBLE_DEVICES=%s  port=%d  log=/tmp/gpu-job-%s.log",
             pid, gpu_list.c_str(), port, job.job_id.c_str());

    if (on_launch_) on_launch_(pid, job);
}

void Launcher::launch_accelerate(JobRequest& job) {
    std::string gpu_list = build_gpu_list(job.assigned_gpus);
    int port = allocate_master_port();
    std::string accelerate_cmd = select_project_command(job, "accelerate");
    std::string mixed_precision = select_accelerate_mixed_precision(job);

    std::vector<std::string> argv = {
        accelerate_cmd, "launch",
        "--num_processes=" + std::to_string(job.num_gpus),
        "--mixed_precision=" + mixed_precision,
        "--main_process_port=" + std::to_string(port),
        job.script
    };
    for (const auto& arg : job.args) argv.push_back(arg);

    std::vector<std::pair<std::string,std::string>> env = {
        {"CUDA_VISIBLE_DEVICES", gpu_list}
    };

    pid_t pid = do_fork_exec(job, argv, env);
    state_.mark_busy(job.assigned_gpus, pid, job.job_id, job.min_vram);
    job.pids.push_back(pid);
    job.started_at = time(nullptr);
    job.master_port = port;
    job.status = JobRequest::RUNNING;

    LOG_INFO("LAUNCH", "accelerate PID=%d  CUDA_VISIBLE_DEVICES=%s  port=%d  log=/tmp/gpu-job-%s.log",
             pid, gpu_list.c_str(), port, job.job_id.c_str());

    if (on_launch_) on_launch_(pid, job);
}

void Launcher::launch_python(JobRequest& job) {
    std::string gpu_list = build_gpu_list(job.assigned_gpus);
    std::string python_cmd = select_python_command(job);

    std::vector<std::string> argv = {python_cmd, job.script};
    for (const auto& arg : job.args) argv.push_back(arg);

    std::vector<std::pair<std::string,std::string>> env = {
        {"CUDA_VISIBLE_DEVICES", gpu_list}
    };

    pid_t pid = do_fork_exec(job, argv, env);
    state_.mark_busy(job.assigned_gpus, pid, job.job_id, job.min_vram);
    job.pids.push_back(pid);
    job.started_at = time(nullptr);
    job.master_port = 0;
    job.status = JobRequest::RUNNING;

    LOG_INFO("LAUNCH", "%s PID=%d  CUDA_VISIBLE_DEVICES=%s  log=/tmp/gpu-job-%s.log",
             python_cmd.c_str(), pid, gpu_list.c_str(), job.job_id.c_str());

    if (on_launch_) on_launch_(pid, job);
}

void Launcher::launch_custom(JobRequest& job) {
    if (job.script.empty()) {
        throw std::runtime_error("custom job has no script/command");
    }

    std::string gpu_list = build_gpu_list(job.assigned_gpus);

    // For custom: script is the command, args are its arguments
    std::vector<std::string> argv = {job.script};
    for (const auto& arg : job.args) argv.push_back(arg);

    std::vector<std::pair<std::string,std::string>> env = {
        {"CUDA_VISIBLE_DEVICES", gpu_list}
    };

    pid_t pid = do_fork_exec(job, argv, env);
    state_.mark_busy(job.assigned_gpus, pid, job.job_id, job.min_vram);
    job.pids.push_back(pid);
    job.started_at = time(nullptr);
    job.master_port = 0;
    job.status = JobRequest::RUNNING;

    LOG_INFO("LAUNCH", "custom PID=%d  CUDA_VISIBLE_DEVICES=%s  log=/tmp/gpu-job-%s.log",
             pid, gpu_list.c_str(), job.job_id.c_str());

    if (on_launch_) on_launch_(pid, job);
}

// ─── HuggingFace Trainer ──────────────────────────────────────────────────
// The Trainer has no launcher of its own — it detects distributed env vars
// set by torchrun. We use --standalone for clean single-node setup.
// Trainer reads: LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
// (all injected by torchrun into each worker subprocess automatically).
void Launcher::launch_hf_trainer(JobRequest& job) {
    std::string gpu_list = build_gpu_list(job.assigned_gpus);
    int port = allocate_master_port();
    std::string torchrun_cmd = select_project_command(job, "torchrun");

    std::vector<std::string> argv = {
        torchrun_cmd,
        "--standalone",                                         // single-node, self-contained rendezvous
        "--nnodes=1",
        "--nproc-per-node=" + std::to_string(job.num_gpus),
        "--master-port=" + std::to_string(port),
        job.script
    };
    for (const auto& arg : job.args) argv.push_back(arg);

    // CUDA_VISIBLE_DEVICES restricts which physical GPUs torchrun can see.
    // torchrun will then assign LOCAL_RANK 0..N-1 across those N GPUs.
    std::vector<std::pair<std::string,std::string>> env = {
        {"CUDA_VISIBLE_DEVICES", gpu_list}
    };

    pid_t pid = do_fork_exec(job, argv, env);
    state_.mark_busy(job.assigned_gpus, pid, job.job_id, job.min_vram);
    job.pids.push_back(pid);
    job.started_at = time(nullptr);
    job.master_port = port;
    job.status = JobRequest::RUNNING;

    LOG_INFO("LAUNCH", "hf_trainer (torchrun --standalone) PID=%d  CUDA_VISIBLE_DEVICES=%s  "
             "nproc=%d  port=%d  log=/tmp/gpu-job-%s.log",
             pid, gpu_list.c_str(), job.num_gpus, port, job.job_id.c_str());

    if (on_launch_) on_launch_(pid, job);
}

// ─── Ray Train ────────────────────────────────────────────────────────────
// Ray Train uses a Python API (TorchTrainer / ScalingConfig). The script
// calls ray.init() + trainer.fit() which spawns Ray actor workers internally.
// We restrict visible GPUs via CUDA_VISIBLE_DEVICES so Ray only schedules
// on the GPUs we've allocated. Ray reads this and isolates each worker.
//
// Key env vars Ray Train sets inside each worker automatically:
//   RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, CUDA_VISIBLE_DEVICES
void Launcher::launch_ray_train(JobRequest& job) {
    std::string gpu_list = build_gpu_list(job.assigned_gpus);
    std::string python_cmd = select_python_command(job);

    std::vector<std::string> argv = {python_cmd, job.script};
    for (const auto& arg : job.args) argv.push_back(arg);

    std::vector<std::pair<std::string,std::string>> env = {
        {"CUDA_VISIBLE_DEVICES",            gpu_list},
        // Tell Ray how many GPUs it can use (matches our allocation)
        {"RAY_NUM_GPUS",                    std::to_string(job.num_gpus)},
        // Disable the "not a Ray cluster" warning when using local mode
        {"RAY_DISABLE_IMPORT_WARNING",      "1"},
        // Suppress Ray's verbose startup banner
        {"RAY_AIR_NEW_OUTPUT",              "1"},
    };

    pid_t pid = do_fork_exec(job, argv, env);
    state_.mark_busy(job.assigned_gpus, pid, job.job_id, job.min_vram);
    job.pids.push_back(pid);
    job.started_at = time(nullptr);
    job.master_port = 0;
    job.status = JobRequest::RUNNING;

    LOG_INFO("LAUNCH", "ray_train PID=%d  CUDA_VISIBLE_DEVICES=%s  num_gpus=%d  "
             "log=/tmp/gpu-job-%s.log",
             pid, gpu_list.c_str(), job.num_gpus, job.job_id.c_str());

    if (on_launch_) on_launch_(pid, job);
}

// ─── Ray Job Submit ───────────────────────────────────────────────────────
// Submits a job to an existing Ray cluster via the Jobs API.
// The scheduler allocates GPUs, sets CUDA_VISIBLE_DEVICES in the runtime env,
// then calls: ray job submit --address <addr> --working-dir <dir>
//                            --runtime-env-json <json> -- python script.py [args]
//
// The Ray head node must already be running: ray start --head
void Launcher::launch_ray_job(JobRequest& job) {
    std::string gpu_list = build_gpu_list(job.assigned_gpus);
    std::string python_cmd = select_python_command();

    // Build --runtime-env-json to pass CUDA_VISIBLE_DEVICES into the Ray job
    std::string runtime_env =
        "{\"env_vars\":{\"CUDA_VISIBLE_DEVICES\":\"" + gpu_list + "\","
        "\"RAY_NUM_GPUS\":\"" + std::to_string(job.num_gpus) + "\"}}";

    // Unique submission ID derived from job_id
    std::string submission_id = "gpu-sched-" + job.job_id;

    std::string working_dir = job.working_dir.empty() ? "." : job.working_dir;

    std::vector<std::string> argv = {
        "ray", "job", "submit",
        "--address",             job.ray_address,
        "--submission-id",       submission_id,
        "--working-dir",         working_dir,
        "--runtime-env-json",    runtime_env,
        "--no-wait",             // return immediately; our monitor tracks by PID of ray CLI
        "--",
        python_cmd, job.script
    };
    for (const auto& arg : job.args) argv.push_back(arg);

    // No CUDA restriction on the ray CLI process itself (it just talks HTTP to the cluster)
    std::vector<std::pair<std::string,std::string>> env = {};

    pid_t pid = do_fork_exec(job, argv, env);
    state_.mark_busy(job.assigned_gpus, pid, job.job_id, job.min_vram);
    job.pids.push_back(pid);
    job.started_at = time(nullptr);
    job.master_port = 0;
    job.status = JobRequest::RUNNING;

    LOG_INFO("LAUNCH", "ray_job  address=%s  submission_id=%s  CUDA_VISIBLE_DEVICES=%s  "
             "log=/tmp/gpu-job-%s.log",
             job.ray_address.c_str(), submission_id.c_str(),
             gpu_list.c_str(), job.job_id.c_str());

    if (on_launch_) on_launch_(pid, job);
}
