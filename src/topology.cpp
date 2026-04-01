#include "topology.h"
#include "logger.h"

#ifdef USE_MOCK_NVML
#  include "mock_nvml.h"
#else
#  include <nvml.h>
#endif

#include <algorithm>
#include <climits>
#include <functional>
#include <stdexcept>

#define NVML_CHECK_TOPO(call) do {                                          \
    nvmlReturn_t _r = (call);                                               \
    if (_r != NVML_SUCCESS) {                                               \
        LOG_WARN("TOPO", "%s failed: %s", #call, nvmlErrorString(_r));      \
    }                                                                       \
} while(0)

static LinkType nvml_topo_to_link(nvmlGpuTopologyLevel_t level) {
    switch (level) {
        case NVML_TOPOLOGY_INTERNAL:   return LinkType::UNKNOWN;    // same GPU
        case NVML_TOPOLOGY_SINGLE:     return LinkType::NVLINK;     // NVLink x1
        case NVML_TOPOLOGY_MULTIPLE:   return LinkType::NVLINK;     // NVLink x2+
        case NVML_TOPOLOGY_HOSTBRIDGE: return LinkType::PCIE_PEER;  // same PCIe switch
        case NVML_TOPOLOGY_NODE:       return LinkType::PCIE_CPU;   // same NUMA, diff switch
        case NVML_TOPOLOGY_SYSTEM:     return LinkType::PCIE_CPU;   // cross-socket
        default:                       return LinkType::UNKNOWN;
    }
}

static const char* link_type_str(LinkType lt) {
    switch (lt) {
        case LinkType::NVLINK:    return "NVLink";
        case LinkType::PCIE_PEER: return "PCIe-Peer";
        case LinkType::PCIE_CPU:  return "PCIe-CPU";
        default:                  return "Unknown";
    }
}

void TopologyMatrix::build(const std::vector<GPUDevice>& devices) {
    size_ = (int)devices.size();
    matrix_.assign(size_, std::vector<LinkType>(size_, LinkType::UNKNOWN));

    for (int i = 0; i < size_; i++) {
        matrix_[i][i] = LinkType::UNKNOWN;  // self
        for (int j = i + 1; j < size_; j++) {
            nvmlGpuTopologyLevel_t topo_level;
            nvmlReturn_t r = nvmlDeviceGetTopologyCommonAncestor(
                devices[i].handle, devices[j].handle, &topo_level);

            LinkType lt;
            if (r == NVML_SUCCESS) {
                lt = nvml_topo_to_link(topo_level);
            } else {
                LOG_WARN("TOPO", "Could not get topology for GPU %d <-> GPU %d: %s",
                         i, j, nvmlErrorString(r));
                lt = LinkType::UNKNOWN;
            }

            matrix_[i][j] = lt;
            matrix_[j][i] = lt;

            LOG_INFO("TOPO", "GPU %d <-> GPU %d: %s (score: %d)",
                     i, j, link_type_str(lt), (int)lt);
        }
    }
}

LinkType TopologyMatrix::get_link(int gpu_a, int gpu_b) const {
    if (gpu_a < 0 || gpu_a >= size_ || gpu_b < 0 || gpu_b >= size_)
        return LinkType::UNKNOWN;
    if (gpu_a == gpu_b) return LinkType::UNKNOWN;
    return matrix_[gpu_a][gpu_b];
}

int TopologyMatrix::score_group(const std::vector<int>& gpu_ids) const {
    int total = 0;
    int n = (int)gpu_ids.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            total += (int)get_link(gpu_ids[i], gpu_ids[j]);
        }
    }
    return total;
}

std::vector<int> TopologyMatrix::find_best_group(
    const std::vector<int>& candidates, int count) const
{
    if ((int)candidates.size() < count)
        return {};
    if ((int)candidates.size() == count)
        return candidates;

    // Brute-force all C(n,k) combinations
    std::vector<int> best;
    int best_score = INT_MIN;

    int n = (int)candidates.size();
    // Use bitmask combination for n <= 16
    std::vector<int> indices(count);
    // Generate combinations using Gosper's hack style recursion
    std::function<void(int, int, std::vector<int>&)> combine =
        [&](int start, int remaining, std::vector<int>& current) {
            if (remaining == 0) {
                int s = score_group(current);
                if (s > best_score) {
                    best_score = s;
                    best = current;
                }
                return;
            }
            for (int i = start; i <= n - remaining; i++) {
                current.push_back(candidates[i]);
                combine(i + 1, remaining - 1, current);
                current.pop_back();
            }
        };

    std::vector<int> current;
    current.reserve(count);
    combine(0, count, current);

    return best;
}

void TopologyMatrix::print() const {
    printf("\nTopology Matrix (link scores):\n     ");
    for (int i = 0; i < size_; i++) printf("GPU%-3d", i);
    printf("\n");
    for (int i = 0; i < size_; i++) {
        printf("GPU%d ", i);
        for (int j = 0; j < size_; j++) {
            if (i == j) {
                printf("  -   ");
            } else {
                LinkType lt = matrix_[i][j];
                switch (lt) {
                    case LinkType::NVLINK:    printf("  NVL "); break;
                    case LinkType::PCIE_PEER: printf("  PP  "); break;
                    case LinkType::PCIE_CPU:  printf("  PC  "); break;
                    default:                  printf("  ?   "); break;
                }
            }
        }
        printf("\n");
    }
    printf("(NVL=NVLink, PP=PCIe-Peer, PC=PCIe-CPU)\n\n");
}
