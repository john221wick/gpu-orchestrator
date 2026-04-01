#include "gpu_device.h"
#include "logger.h"

#ifdef USE_MOCK_NVML
#  include "mock_nvml.h"
#else
#  include <nvml.h>
#endif

#include <vector>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <string>

#define NVML_CHECK(call) do {                                               \
    nvmlReturn_t _r = (call);                                               \
    if (_r != NVML_SUCCESS) {                                               \
        LOG_ERROR("NVML", "%s failed: %s at %s:%d",                         \
                  #call, nvmlErrorString(_r), __FILE__, __LINE__);          \
    }                                                                       \
} while(0)

static int get_numa_node(unsigned int pcie_bus) {
    // Try to read NUMA affinity from sysfs
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/0000:%02x:00.0/numa_node", pcie_bus);
    std::ifstream f(path);
    if (f.is_open()) {
        int node = -1;
        f >> node;
        return node;
    }
    return -1;
}

std::vector<GPUDevice> discover_gpus() {
    nvmlReturn_t ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        throw std::runtime_error(std::string("nvmlInit failed: ") + nvmlErrorString(ret));
    }

    unsigned int count = 0;
    NVML_CHECK(nvmlDeviceGetCount(&count));

    if (count == 0) {
        throw std::runtime_error("No GPUs found");
    }

    std::vector<GPUDevice> devices;
    devices.reserve(count);

    for (unsigned int i = 0; i < count; i++) {
        GPUDevice dev;
        memset(&dev, 0, sizeof(dev));
        dev.id = (int)i;

        NVML_CHECK(nvmlDeviceGetHandleByIndex(i, &dev.handle));
        NVML_CHECK(nvmlDeviceGetName(dev.handle, dev.name, sizeof(dev.name)));
        NVML_CHECK(nvmlDeviceGetUUID(dev.handle, dev.uuid, sizeof(dev.uuid)));

        // VRAM
        nvmlMemory_t mem;
        NVML_CHECK(nvmlDeviceGetMemoryInfo(dev.handle, &mem));
        dev.vram_total = (size_t)mem.total;
        dev.vram_free  = (size_t)mem.free;

        // SM count
        NVML_CHECK(nvmlDeviceGetNumGpuCores(dev.handle, &dev.sm_count));

        // Temperature
        NVML_CHECK(nvmlDeviceGetTemperature(dev.handle, 0 /* NVML_TEMPERATURE_GPU */, &dev.temperature));

        // PCIe info
        nvmlPciInfo_t pci;
        NVML_CHECK(nvmlDeviceGetPciInfo(dev.handle, &pci));
        dev.pcie_bus    = pci.bus;
        dev.pcie_device = pci.device;
        dev.numa_node   = get_numa_node(pci.bus);

        // State
        dev.is_busy    = false;
        dev.owner_pid  = 0;

        devices.push_back(dev);

        LOG_INFO("INIT", "GPU %d: %s  UUID=%s  VRAM=%.1f GB  SMs=%u  Temp=%u°C  NUMA=%d",
                 dev.id, dev.name,
                 dev.uuid,
                 (double)dev.vram_total / (1024.0 * 1024.0 * 1024.0),
                 dev.sm_count,
                 dev.temperature,
                 dev.numa_node);
    }

    LOG_INFO("INIT", "Discovered %ux %s", count, devices[0].name);
    return devices;
}
