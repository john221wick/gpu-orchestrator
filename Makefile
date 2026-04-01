.DEFAULT_GOAL := all

CXX ?= c++
CPPFLAGS += -Iinclude -Ithird_party
CXXFLAGS += -std=c++17 -Wall -Wextra -O2 -pthread
LDLIBS += -lpthread

NVML_INCLUDE_CANDIDATES := \
	$(strip $(NVML_INCLUDE_DIR)) \
	$(strip $(CUDA_HOME))/include \
	$(strip $(CUDA_PATH))/include \
	/usr/include \
	/usr/include/nvidia/gdk \
	/usr/local/cuda/include \
	/opt/cuda/include

NVML_LIB_CANDIDATES := \
	$(strip $(NVML_LIB_DIR)) \
	$(strip $(CUDA_HOME))/lib64 \
	$(strip $(CUDA_HOME))/targets/x86_64-linux/lib \
	$(strip $(CUDA_PATH))/lib64 \
	/usr/lib64 \
	/usr/lib/x86_64-linux-gnu \
	/usr/lib/wsl/lib \
	/usr/local/cuda/lib64 \
	/opt/cuda/lib64 \
	/usr/lib

NVML_HEADER := $(firstword $(foreach dir,$(NVML_INCLUDE_CANDIDATES),$(wildcard $(dir)/nvml.h)))
NVML_HEADER_DIR := $(patsubst %/nvml.h,%,$(NVML_HEADER))

NVML_LIBRARY := $(firstword $(foreach dir,$(NVML_LIB_CANDIDATES),$(wildcard $(dir)/libnvidia-ml.so) $(wildcard $(dir)/libnvidia-ml.so.1)))
NVML_LIBRARY_DIR := $(patsubst %/libnvidia-ml.so,%,$(patsubst %/libnvidia-ml.so.1,%,$(NVML_LIBRARY)))

ifneq ($(strip $(NVML_HEADER_DIR)),)
REAL_CPPFLAGS += -I$(NVML_HEADER_DIR)
endif

ifneq ($(strip $(NVML_LIBRARY_DIR)),)
REAL_LDFLAGS += -L$(NVML_LIBRARY_DIR)
endif

BUILD_DIR := build
COMMON_DIR := $(BUILD_DIR)/common
MOCK_DIR := $(BUILD_DIR)/mock
REAL_DIR := $(BUILD_DIR)/real

SOCKET_PATH ?= /tmp/gpu-scheduler.sock
SCHEDULER_LOG ?= /tmp/gpu-scheduler-daemon.log

JOB_FAST ?= ./scripts/job_demo_fast.json
JOB_SLOW ?= ./scripts/job_demo_slow.json
JOB_QUEUED ?= ./scripts/job_demo_queued.json

HF_FINETUNE_JOB ?= ./scripts/job_hf_personal_chat.json
HF_INFERENCE_JOB ?= ./scripts/job_hf_personal_chat_inference.json
HF_INFERENCE_MODEL ?= Qwen/Qwen3-4B-Instruct-2507
HF_INFERENCE_ADAPTER_PATH ?= outputs/vast-personal-chat
PROMPT ?=

DAEMON_SRCS := \
	src/daemon.cpp \
	src/gpu_discovery.cpp \
	src/topology.cpp \
	src/job_queue.cpp \
	src/gpu_state.cpp \
	src/scheduler.cpp \
	src/launcher.cpp \
	src/process_monitor.cpp \
	src/server.cpp \
	src/logger.cpp

GPU_SUBMIT_SRCS := tools/gpu-submit.cpp
GPU_STATUS_SRCS := tools/gpu-status.cpp src/logger.cpp

TEST_TOPOLOGY_SRCS := test/test_topology.cpp src/topology.cpp src/logger.cpp
TEST_QUEUE_SRCS := test/test_queue.cpp src/job_queue.cpp src/logger.cpp
TEST_SCHEDULER_SRCS := \
	test/test_scheduler.cpp \
	src/scheduler.cpp \
	src/gpu_state.cpp \
	src/topology.cpp \
	src/job_queue.cpp \
	src/launcher.cpp \
	src/logger.cpp

COMMON_SUBMIT_OBJS := $(patsubst %.cpp,$(COMMON_DIR)/%.o,$(GPU_SUBMIT_SRCS))
COMMON_STATUS_OBJS := $(patsubst %.cpp,$(COMMON_DIR)/%.o,$(GPU_STATUS_SRCS))
MOCK_DAEMON_OBJS := $(patsubst %.cpp,$(MOCK_DIR)/%.o,$(DAEMON_SRCS))
REAL_DAEMON_OBJS := $(patsubst %.cpp,$(REAL_DIR)/%.o,$(DAEMON_SRCS))
TEST_TOPOLOGY_OBJS := $(patsubst %.cpp,$(MOCK_DIR)/%.o,$(TEST_TOPOLOGY_SRCS))
TEST_QUEUE_OBJS := $(patsubst %.cpp,$(MOCK_DIR)/%.o,$(TEST_QUEUE_SRCS))
TEST_SCHEDULER_OBJS := $(patsubst %.cpp,$(MOCK_DIR)/%.o,$(TEST_SCHEDULER_SRCS))

ALL_OBJS := $(sort \
	$(COMMON_SUBMIT_OBJS) \
	$(COMMON_STATUS_OBJS) \
	$(MOCK_DAEMON_OBJS) \
	$(REAL_DAEMON_OBJS) \
	$(TEST_TOPOLOGY_OBJS) \
	$(TEST_QUEUE_OBJS) \
	$(TEST_SCHEDULER_OBJS))

ALL_DEPS := $(ALL_OBJS:.o=.d)

.PHONY: all mock real tools test test-mock run run-real start real-start \
	status scheduler-status submit-fast submit-slow submit-queued demo \
	vast-start ensure-real ensure-scheduler submit-hf-personal-chat \
	submit-hf-inference hf-finetune hf-fiinetune hf-inference \
	install clean clean-runtime help check-real

all: mock
	@echo ""
	@echo "Built default development binaries:"
	@echo "  ./gpu-scheduler-mock"
	@echo "  ./gpu-submit"
	@echo "  ./gpu-status"
	@echo ""
	@echo "Use 'make real' on a Linux/NVIDIA host with NVML installed."

mock: gpu-scheduler-mock gpu-submit gpu-status

real: check-real gpu-scheduler gpu-submit gpu-status
	@echo ""
	@echo "Built production binaries:"
	@echo "  ./gpu-scheduler"
	@echo "  ./gpu-submit"
	@echo "  ./gpu-status"

tools: gpu-submit gpu-status

check-real:
	@if [ -z "$(NVML_HEADER_DIR)" ]; then \
		echo "NVML headers not found." >&2; \
		echo "Set NVML_INCLUDE_DIR=/path/to/include if nvml.h is installed in a non-standard location." >&2; \
		exit 1; \
	fi
	@if [ -z "$(NVML_LIBRARY_DIR)" ] && [ -z "$(NVML_LIB_DIR)" ]; then \
		echo "NVML library not found in common locations." >&2; \
		echo "Set NVML_LIB_DIR=/path/to/lib if libnvidia-ml is installed in a non-standard location." >&2; \
		exit 1; \
	fi

gpu-scheduler: $(REAL_DAEMON_OBJS)
	$(CXX) $(LDFLAGS) $(REAL_LDFLAGS) -o $@ $^ $(LDLIBS) -lnvidia-ml

gpu-scheduler-mock: $(MOCK_DAEMON_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

gpu-submit: $(COMMON_SUBMIT_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

gpu-status: $(COMMON_STATUS_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test-topology: $(TEST_TOPOLOGY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test-queue: $(TEST_QUEUE_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test-scheduler: $(TEST_SCHEDULER_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test: test-topology test-queue test-scheduler
	./test-topology
	./test-queue
	./test-scheduler

test-mock: test

run: mock
	./gpu-scheduler-mock --socket "$(SOCKET_PATH)" --log "$(SCHEDULER_LOG)"

run-real: real
	./gpu-scheduler --socket "$(SOCKET_PATH)" --log "$(SCHEDULER_LOG)"

start: mock gpu-status
	@if [ -S "$(SOCKET_PATH)" ] && ./gpu-status --socket "$(SOCKET_PATH)" >/dev/null 2>&1; then \
		echo "[gpu-orchestrator] Scheduler already running on $(SOCKET_PATH)"; \
	else \
		rm -f "$(SOCKET_PATH)"; \
		nohup ./gpu-scheduler-mock --socket "$(SOCKET_PATH)" --log "$(SCHEDULER_LOG)" >/dev/null 2>&1 & \
		pid=$$!; \
		for attempt in 1 2 3 4 5; do \
			if [ -S "$(SOCKET_PATH)" ]; then \
				echo "[gpu-orchestrator] gpu-scheduler-mock started"; \
				echo "[gpu-orchestrator] pid=$$pid"; \
				echo "[gpu-orchestrator] socket=$(SOCKET_PATH)"; \
				echo "[gpu-orchestrator] log=$(SCHEDULER_LOG)"; \
				exit 0; \
			fi; \
			sleep 1; \
		done; \
		echo "[gpu-orchestrator] Scheduler failed to create $(SOCKET_PATH)" >&2; \
		echo "[gpu-orchestrator] Check $(SCHEDULER_LOG) for details." >&2; \
		exit 1; \
	fi

real-start: real gpu-status
	GPU_SCHEDULER_SOCKET="$(SOCKET_PATH)" GPU_SCHEDULER_LOG="$(SCHEDULER_LOG)" ./scripts/vast_start_scheduler.sh

vast-start: real-start

ensure-real: real gpu-status
	@if ./gpu-status --socket "$(SOCKET_PATH)" >/dev/null 2>&1; then \
		echo "[gpu-orchestrator] Scheduler already running on $(SOCKET_PATH)"; \
	else \
		$(MAKE) real-start SOCKET_PATH="$(SOCKET_PATH)" SCHEDULER_LOG="$(SCHEDULER_LOG)"; \
	fi

ensure-scheduler: ensure-real

status: gpu-status
	./gpu-status --socket "$(SOCKET_PATH)"

scheduler-status: status

submit-fast: gpu-submit
	./gpu-submit "$(JOB_FAST)" --socket "$(SOCKET_PATH)"

submit-slow: gpu-submit
	./gpu-submit "$(JOB_SLOW)" --socket "$(SOCKET_PATH)"

submit-queued: gpu-submit
	./gpu-submit "$(JOB_QUEUED)" --socket "$(SOCKET_PATH)"

demo: start submit-fast
	@$(MAKE) status SOCKET_PATH="$(SOCKET_PATH)"

submit-hf-personal-chat: gpu-submit
	GPU_SCHEDULER_SOCKET="$(SOCKET_PATH)" ./scripts/vast_submit_hf_personal_chat.sh "$(HF_FINETUNE_JOB)"

submit-hf-inference: gpu-submit
	@if [ -n "$(PROMPT)" ] || [ "$(HF_INFERENCE_MODEL)" != "Qwen/Qwen3-4B-Instruct-2507" ] || [ "$(HF_INFERENCE_ADAPTER_PATH)" != "outputs/vast-personal-chat" ]; then \
		GPU_SCHEDULER_SOCKET="$(SOCKET_PATH)" HF_INFERENCE_MODEL="$(HF_INFERENCE_MODEL)" HF_INFERENCE_ADAPTER_PATH="$(HF_INFERENCE_ADAPTER_PATH)" HF_INFERENCE_PROMPT="$(PROMPT)" ./scripts/vast_submit_hf_inference.sh; \
	else \
		GPU_SCHEDULER_SOCKET="$(SOCKET_PATH)" ./scripts/vast_submit_hf_inference.sh "$(HF_INFERENCE_JOB)"; \
	fi

hf-finetune: ensure-real submit-hf-personal-chat
	@$(MAKE) status SOCKET_PATH="$(SOCKET_PATH)"

hf-fiinetune: hf-finetune

hf-inference: ensure-real submit-hf-inference
	@$(MAKE) status SOCKET_PATH="$(SOCKET_PATH)"

install: real
	install -m 755 gpu-scheduler /usr/local/bin/gpu-scheduler
	install -m 755 gpu-submit /usr/local/bin/gpu-submit
	install -m 755 gpu-status /usr/local/bin/gpu-status

clean:
	rm -rf build gpu-scheduler gpu-scheduler-mock gpu-submit gpu-status
	rm -f test-topology test-queue test-scheduler

clean-runtime:
	rm -f "$(SOCKET_PATH)" "$(SCHEDULER_LOG)" /tmp/gpu-job-*.log

help:
	@echo "Targets:"
	@echo "  make           Build the default mock scheduler and CLI tools"
	@echo "  make real      Build the NVML-backed scheduler and CLI tools"
	@echo "  make test      Build and run unit tests"
	@echo "  make run       Run gpu-scheduler-mock in the foreground"
	@echo "  make start     Start gpu-scheduler-mock in the background"
	@echo "  make status    Show scheduler status"
	@echo "  make submit-fast    Submit scripts/job_demo_fast.json"
	@echo "  make submit-slow    Submit scripts/job_demo_slow.json"
	@echo "  make submit-queued  Submit scripts/job_demo_queued.json"
	@echo "  make real-start     Start the real scheduler helper script"
	@echo "  make hf-finetune    Submit the finetune example job"
	@echo "  make hf-inference   Submit the inference example job"
	@echo "  make clean          Remove build output"
	@echo "  make clean-runtime  Remove socket and log files"
	@echo ""
	@echo "Variables:"
	@echo "  SOCKET_PATH=$(SOCKET_PATH)"
	@echo "  SCHEDULER_LOG=$(SCHEDULER_LOG)"
	@echo "  NVML_INCLUDE_DIR=$(NVML_INCLUDE_DIR)"
	@echo "  NVML_LIB_DIR=$(NVML_LIB_DIR)"
	@echo "  detected NVML header dir=$(NVML_HEADER_DIR)"
	@echo "  detected NVML lib dir=$(NVML_LIBRARY_DIR)"

$(COMMON_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(MOCK_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DUSE_MOCK_NVML -MMD -MP -c $< -o $@

$(REAL_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(REAL_CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(ALL_DEPS)
