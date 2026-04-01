#pragma once
#include <string>
#include <vector>
#include <ctime>
#include <unistd.h>

enum class Framework {
    TORCHRUN,           // torchrun --standalone --nproc-per-node=N script.py
    DEEPSPEED,          // deepspeed --num_gpus=N script.py
    ACCELERATE,         // accelerate launch --num_processes=N script.py
    PYTHON,             // python script.py (single GPU)
    CUSTOM,             // exec arbitrary command
    HF_TRAINER,         // HuggingFace Trainer: torchrun --standalone + Trainer detects env
    RAY_TRAIN,          // Ray Train: python script.py — Ray spawns workers internally
    RAY_JOB,            // ray job submit --address <ray_address> -- python script.py
};

enum class JobType {
    TRAINING,       // long-running, multi-GPU, high VRAM
    INFERENCE,      // usually single GPU
    FINETUNING      // like training but typically shorter
};

struct JobRequest {
    std::string                 job_id;
    std::string                 user_id;
    Framework                   framework = Framework::PYTHON;
    JobType                     job_type = JobType::INFERENCE;
    int                         num_gpus = 1;
    size_t                      min_vram = 0;       // bytes per GPU
    bool                        needs_peer = false;
    int                         priority = 5;       // 0 = highest
    time_t                      submitted_at = 0;

    std::string                 script;
    std::vector<std::string>    args;
    std::string                 working_dir;

    int                         max_time_sec = 0;   // 0 = no limit

    // RAY_JOB only: Ray dashboard address (default: http://localhost:8265)
    std::string                 ray_address = "http://localhost:8265";

    enum Status { PENDING, RUNNING, COMPLETED, FAILED, KILLED, QUEUED };
    Status                      status = PENDING;
    std::vector<int>            assigned_gpus;
    std::vector<pid_t>          pids;
    time_t                      started_at = 0;
    time_t                      finished_at = 0;
    int                         master_port = 0;
    int                         exit_code = -1;
};

inline const char* framework_to_str(Framework f) {
    switch (f) {
        case Framework::TORCHRUN:    return "torchrun";
        case Framework::DEEPSPEED:   return "deepspeed";
        case Framework::ACCELERATE:  return "accelerate";
        case Framework::PYTHON:      return "python";
        case Framework::CUSTOM:      return "custom";
        case Framework::HF_TRAINER:  return "hf_trainer";
        case Framework::RAY_TRAIN:   return "ray_train";
        case Framework::RAY_JOB:     return "ray_job";
    }
    return "unknown";
}

inline Framework str_to_framework(const std::string& s) {
    if (s == "torchrun")             return Framework::TORCHRUN;
    if (s == "deepspeed")            return Framework::DEEPSPEED;
    if (s == "accelerate")           return Framework::ACCELERATE;
    if (s == "custom")               return Framework::CUSTOM;
    if (s == "hf_trainer" ||
        s == "trainer" ||
        s == "huggingface")          return Framework::HF_TRAINER;
    if (s == "ray_train" ||
        s == "ray-train")            return Framework::RAY_TRAIN;
    if (s == "ray_job"  ||
        s == "ray-job"  ||
        s == "ray")                  return Framework::RAY_JOB;
    return Framework::PYTHON;
}

inline const char* jobtype_to_str(JobType t) {
    switch (t) {
        case JobType::TRAINING:   return "training";
        case JobType::INFERENCE:  return "inference";
        case JobType::FINETUNING: return "finetuning";
    }
    return "unknown";
}

inline JobType str_to_jobtype(const std::string& s) {
    if (s == "training")   return JobType::TRAINING;
    if (s == "finetuning") return JobType::FINETUNING;
    return JobType::INFERENCE;
}
