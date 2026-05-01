# TinyTorch Makefile
# Builds tests against the full framework source tree.
# Usage:
#   make all-tier1       # self-contained tests (wrappers + autograd + integration)
#   make run-tier1       # build + execute all self-contained tests
#   make all             # everything including PyTorch-comparison tests
#   make <test-name>     # single test (e.g. make build/relu-wrapper)
#   make clean

NVCC      := nvcc
ARCH      := -arch=sm_75
NVCCFLAGS := $(ARCH) -O2
LDLIBS    := -lcurand

# Framework sources — every test links against all of these.
CUDA_SRCS := $(wildcard cuda/*.cu)
CPP_SRCS  := include/tensor.cpp \
             $(wildcard src/autograd/*.cpp) \
             $(wildcard src/functional/*.cpp) \
             $(wildcard src/nn/*.cpp) \
             $(wildcard src/optim/*.cpp)
CORE_SRCS := $(CUDA_SRCS) $(CPP_SRCS)

MNIST_SRCS := mnist-dataloader/mnist.cpp

WRAPPER_TESTS  := relu-wrapper linear-wrapper linear-bias-wrapper conv-wrapper \
                  global-pooling-wrapper cross-entropy-wrapper sgd-wrapper
AUTOGRAD_TESTS := autograd-test exhaustive-autograd-test

# Single binary: longest practical graph forward + backward(loss) (Conv->ReLU->Pool->Linear(bias)->CE)
INTEGRATION_TESTS := full-pipeline-test

BUILD_DIR := build

# --- Per-test build rules ---------------------------------------------------

$(BUILD_DIR)/%-wrapper: tests/wrapper/%-wrapper.cpp $(CORE_SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(CORE_SRCS) $(LDLIBS) -o $@

$(BUILD_DIR)/autograd-test: tests/autograd/autograd-test.cpp $(CORE_SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(CORE_SRCS) $(LDLIBS) -o $@

$(BUILD_DIR)/exhaustive-autograd-test: tests/autograd/exhaustive-autograd-test.cpp $(CORE_SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(CORE_SRCS) $(LDLIBS) -o $@

$(BUILD_DIR)/full-pipeline-test: tests/integration/full-pipeline-test.cpp $(CORE_SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(CORE_SRCS) $(LDLIBS) -o $@

$(BUILD_DIR)/test_matmul: tests/matmul/test_matmul.c $(CORE_SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x cu $< $(CORE_SRCS) $(LDLIBS) -o $@

$(BUILD_DIR)/test_conv: tests/conv/test_conv.c $(CORE_SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x cu $< $(CORE_SRCS) $(LDLIBS) -o $@

$(BUILD_DIR)/model-def: tests/model-definition/model-def.cpp $(CORE_SRCS) $(MNIST_SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(CORE_SRCS) $(MNIST_SRCS) $(LDLIBS) -o $@

# --- Aggregate targets ------------------------------------------------------

all-wrappers: $(addprefix $(BUILD_DIR)/, $(WRAPPER_TESTS))
all-autograd: $(addprefix $(BUILD_DIR)/, $(AUTOGRAD_TESTS))
all-integration: $(addprefix $(BUILD_DIR)/, $(INTEGRATION_TESTS))
all-tier1:    all-wrappers all-autograd all-integration
all:          all-tier1 $(BUILD_DIR)/test_matmul $(BUILD_DIR)/test_conv $(BUILD_DIR)/model-def

# Run every self-contained test sequentially.
run-tier1: all-tier1
	@for t in $(WRAPPER_TESTS) $(AUTOGRAD_TESTS) $(INTEGRATION_TESTS); do \
	    echo "=== $$t ==="; \
	    ./$(BUILD_DIR)/$$t || exit 1; \
	    echo ""; \
	done

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all all-tier1 all-wrappers all-autograd all-integration run-tier1 clean
