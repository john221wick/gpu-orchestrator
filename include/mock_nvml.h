#pragma once
// Mock NVML layer for development/testing without a physical GPU.
// Activated with -DUSE_MOCK_NVML at compile time.
// Simulates 4 GPUs: GPU0-GPU1 NVLink, GPU2-GPU3 NVLink, others PCIe.

#ifdef USE_MOCK_NVML

#include <cstdint>
#include <cstring>
#include <cstdio>

// ---- Types from real nvml.h ----
typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;

#define NVML_SUCCESS             0
#define NVML_ERROR_UNINITIALIZED 1
#define NVML_ERROR_NOT_FOUND     6

typedef enum {
    NVML_TOPOLOGY_INTERNAL     = 0,
    NVML_TOPOLOGY_SINGLE       = 1,
    NVML_TOPOLOGY_MULTIPLE     = 2,
    NVML_TOPOLOGY_HOSTBRIDGE   = 3,
    NVML_TOPOLOGY_NODE         = 4,
    NVML_TOPOLOGY_SYSTEM       = 5,
} nvmlGpuTopologyLevel_t;

typedef enum {
    NVML_TEMPERATURE_GPU = 0,
} nvmlTemperatureSensors_t;

typedef struct {
    unsigned int bus;
    unsigned int device;
    unsigned int domain;
    unsigned int busId;
} nvmlPciInfo_t;

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

#define NVML_DEVICE_NAME_BUFFER_SIZE  64
#define NVML_DEVICE_UUID_BUFFER_SIZE  96

inline const char* nvmlErrorString(nvmlReturn_t) { return "mock_error"; }

// ---- Mock state ----
static const int MOCK_GPU_COUNT = 4;
static const char* MOCK_NAMES[4] = {
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100-SXM4-40GB"
};
static const char* MOCK_UUIDS[4] = {
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-11111111-1111-1111-1111-111111111111",
    "GPU-22222222-2222-2222-2222-222222222222",
    "GPU-33333333-3333-3333-3333-333333333333"
};

// Topology: 0<->1 NVLink, 2<->3 NVLink, everything else PCIe-CPU
static nvmlGpuTopologyLevel_t mock_topo[4][4] = {
    { NVML_TOPOLOGY_INTERNAL,   NVML_TOPOLOGY_SINGLE,    NVML_TOPOLOGY_NODE,      NVML_TOPOLOGY_NODE      },
    { NVML_TOPOLOGY_SINGLE,     NVML_TOPOLOGY_INTERNAL,  NVML_TOPOLOGY_NODE,      NVML_TOPOLOGY_NODE      },
    { NVML_TOPOLOGY_NODE,       NVML_TOPOLOGY_NODE,      NVML_TOPOLOGY_INTERNAL,  NVML_TOPOLOGY_SINGLE    },
    { NVML_TOPOLOGY_NODE,       NVML_TOPOLOGY_NODE,      NVML_TOPOLOGY_SINGLE,    NVML_TOPOLOGY_INTERNAL  },
};

// ---- Mock API implementations ----
inline nvmlReturn_t nvmlInit() { return NVML_SUCCESS; }
inline nvmlReturn_t nvmlShutdown() { return NVML_SUCCESS; }

inline nvmlReturn_t nvmlDeviceGetCount(unsigned int* count) {
    *count = MOCK_GPU_COUNT;
    return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
    if (index >= (unsigned)MOCK_GPU_COUNT) return NVML_ERROR_NOT_FOUND;
    *device = (nvmlDevice_t)(uintptr_t)(index + 1);  // non-null handle
    return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int len) {
    int idx = (int)(uintptr_t)device - 1;
    strncpy(name, MOCK_NAMES[idx], len);
    return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int len) {
    int idx = (int)(uintptr_t)device - 1;
    strncpy(uuid, MOCK_UUIDS[idx], len);
    return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* mem) {
    (void)device;
    mem->total = (size_t)40 * 1024 * 1024 * 1024ULL;  // 40 GB
    mem->free  = (size_t)40 * 1024 * 1024 * 1024ULL;
    mem->used  = 0;
    return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t, unsigned int* cores) {
    *cores = 108; // A100 SM count
    return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetTemperature(
    nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int* temp)
{
    *temp = 40;
    return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci) {
    int idx = (int)(uintptr_t)device - 1;
    pci->bus = (unsigned)(idx * 16);
    pci->device = (unsigned)idx;
    pci->domain = 0;
    pci->busId = (unsigned)(idx * 16);
    return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(
    nvmlDevice_t dev1, nvmlDevice_t dev2, nvmlGpuTopologyLevel_t* topo)
{
    int i = (int)(uintptr_t)dev1 - 1;
    int j = (int)(uintptr_t)dev2 - 1;
    if (i < 0 || i >= MOCK_GPU_COUNT || j < 0 || j >= MOCK_GPU_COUNT)
        return NVML_ERROR_NOT_FOUND;
    *topo = mock_topo[i][j];
    return NVML_SUCCESS;
}

#else
// Real NVML: include the actual header
#include <nvml.h>
#endif // USE_MOCK_NVML
