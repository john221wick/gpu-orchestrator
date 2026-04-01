#pragma once
#ifdef USE_MOCK_NVML
#  include "mock_nvml.h"
#else
#  include <nvml.h>
#endif

#include <string>
#include <cstdint>
#include <unistd.h>

struct GPUDevice {
    int             id;                 // 0-indexed GPU ID
    char            name[64];           // e.g., "NVIDIA A100-SXM4-40GB"
    char            uuid[96];           // unique identifier
    size_t          vram_total;         // total VRAM in bytes
    size_t          vram_free;          // current free VRAM in bytes
    unsigned int    sm_count;           // streaming multiprocessor count
    unsigned int    temperature;        // current temp in Celsius

    // PCIe topology info
    unsigned int    pcie_bus;
    unsigned int    pcie_device;
    int             numa_node;          // closest CPU NUMA node

    // State
    bool            is_busy;            // is a job assigned?
    pid_t           owner_pid;          // PID of owning job (0 if free)
    std::string     owner_job_id;       // job ID using this GPU

    // NVML handle for live queries
    nvmlDevice_t    handle;
};
