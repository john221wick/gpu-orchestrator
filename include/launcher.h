#pragma once
#include "job.h"
#include "gpu_state.h"
#include <functional>
#include <mutex>
#include <set>

class Launcher {
public:
    using OnLaunch = std::function<void(pid_t pid, const JobRequest& job)>;

    Launcher(GPUStateTracker& state, OnLaunch on_launch);

    // Launch a job on its assigned GPUs.
    // Sets env vars, fork()s, exec()s the framework command.
    void launch(JobRequest& job);

    // Release a master port when a job completes
    void release_port(int port);

private:
    void launch_torchrun(JobRequest& job);
    void launch_deepspeed(JobRequest& job);
    void launch_accelerate(JobRequest& job);
    void launch_python(JobRequest& job);
    void launch_custom(JobRequest& job);
    void launch_hf_trainer(JobRequest& job);
    void launch_ray_train(JobRequest& job);
    void launch_ray_job(JobRequest& job);

    // Build CUDA_VISIBLE_DEVICES string
    std::string build_gpu_list(const std::vector<int>& gpus);

    // Allocate a unique master port
    int allocate_master_port();

    // Core fork/exec: sets env, chdir, redirects stdout/stderr, exec's argv
    pid_t do_fork_exec(const JobRequest& job, const std::vector<std::string>& argv,
                       const std::vector<std::pair<std::string,std::string>>& env_extras);

    GPUStateTracker& state_;
    OnLaunch         on_launch_;
    std::mutex       port_mutex_;
    std::set<int>    used_ports_;
    int              next_port_ = 29500;
};
