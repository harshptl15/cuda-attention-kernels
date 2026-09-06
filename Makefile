# Local build for a CUDA-capable machine.
# (For a hosted GPU, notebooks/cuda_kernels_colab.ipynb runs the same kernels.)
#
# Set ARCH to the target GPU's compute capability:
#   T4 = sm_75, A100 = sm_80, RTX 30xx = sm_86, RTX 40xx = sm_89, H100 = sm_90

NVCC := nvcc
ARCH := sm_75
FLAGS := -O3 -arch=$(ARCH)
SRC_DIR := kernels
BUILD_DIR := build

TARGETS := vector_add matmul_naive matmul_tiled softmax_fused flash_attention

.PHONY: all run clean

all: $(addprefix $(BUILD_DIR)/,$(TARGETS))

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/vector_add: $(SRC_DIR)/01_vector_add.cu | $(BUILD_DIR)
	$(NVCC) $(FLAGS) $< -o $@

$(BUILD_DIR)/matmul_naive: $(SRC_DIR)/02_matmul_naive.cu | $(BUILD_DIR)
	$(NVCC) $(FLAGS) $< -o $@

$(BUILD_DIR)/matmul_tiled: $(SRC_DIR)/03_matmul_tiled.cu | $(BUILD_DIR)
	$(NVCC) $(FLAGS) $< -o $@

$(BUILD_DIR)/softmax_fused: $(SRC_DIR)/04_softmax_fused.cu | $(BUILD_DIR)
	$(NVCC) $(FLAGS) $< -o $@

$(BUILD_DIR)/flash_attention: $(SRC_DIR)/05_flash_attention_simplified.cu | $(BUILD_DIR)
	$(NVCC) $(FLAGS) $< -o $@

run: all
	@for t in $(TARGETS); do \
		echo "=== $$t ==="; \
		./$(BUILD_DIR)/$$t; \
		echo; \
	done

clean:
	rm -rf $(BUILD_DIR)
