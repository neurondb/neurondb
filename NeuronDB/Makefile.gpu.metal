#-------------------------------------------------------------------------
# Makefile.gpu.metal
#     Metal GPU backend build configuration (macOS/Apple Silicon)
#
# Copyright (c) 2024-2025, neurondb, Inc.
#-------------------------------------------------------------------------

# Note: Makefile.gpu.common is included by Makefile.core, not here
# This prevents duplicate includes

# Platform check (UNAME_S should be set by Makefile.core)
ifeq ($(UNAME_S),)
UNAME_S := $(shell uname -s)
endif

ifeq ($(UNAME_S),Darwin)
	# Metal is available on macOS
	METAL_AVAILABLE := yes
	
	# Metal source files
	METAL_C_SOURCES := \
		src/gpu/metal/gpu_metal.c \
		src/gpu/metal/gpu_backend_metal.c
	METAL_M_SOURCES := \
		src/gpu/metal/gpu_metal_impl.m
	
	METAL_OBJS := $(METAL_C_SOURCES:.c=.o) $(METAL_M_SOURCES:.m=.o)
	
	# Metal compilation flags
	METAL_CPPFLAGS = -DNDB_GPU_METAL
	METAL_CPPFLAGS += -framework Metal -framework MetalPerformanceShaders
	METAL_CPPFLAGS += -framework Accelerate -framework Foundation
	
	# Metal link flags
	METAL_LDFLAGS = -framework Metal -framework MetalPerformanceShaders
	METAL_LDFLAGS += -framework Accelerate -framework Foundation
	
	# Add to GPU configuration
	GPU_CPPFLAGS += $(METAL_CPPFLAGS)
	GPU_LDFLAGS += $(METAL_LDFLAGS)
	GPU_OBJS += $(METAL_OBJS)
	
	# Add Metal objects to main OBJS (only once)
ifndef METAL_OBJS_ADDED
METAL_OBJS_ADDED := 1
OBJS += $(METAL_OBJS)
endif
	
	# Metal shader compilation (prerequisite)
	METAL_SHADERS = src/gpu/metal/neurondb_gpu_kernels.metallib
	-include Makefile.metal.precompile
	
	# Update GPU_BACKENDS
	ifeq ($(GPU_BACKENDS),none)
		GPU_BACKENDS := metal
	else
		GPU_BACKENDS := $(GPU_BACKENDS) metal
	endif
	
	# Metal shaders must be built before the library (defined after MODULE_big is set)
	# This will be handled in Makefile.core
else
	METAL_AVAILABLE := no
	$(info Metal only available on macOS - Metal backend disabled)
endif

.PHONY: metal-check metal-info

metal-check:
	@echo "Platform: $(UNAME_S)"
	@echo "Metal Available: $(METAL_AVAILABLE)"
	@echo "Metal Sources: $(words $(METAL_SOURCES)) files"
	@echo "Metal Objects: $(words $(METAL_OBJS)) files"
	@echo "Metal Shaders: $(METAL_SHADERS)"

metal-info:
	@echo "══════════════════════════════════════════════════════════════"
	@echo "Metal GPU Backend Configuration"
	@echo "══════════════════════════════════════════════════════════════"
	@echo "Platform: $(UNAME_S)"
	@echo "Status: $(if $(filter metal,$(GPU_BACKENDS)),Enabled,Disabled)"
	@echo "Sources: $(words $(METAL_SOURCES)) files"
	@echo "Objects: $(words $(METAL_OBJS)) files"
	@echo "Shaders: $(METAL_SHADERS)"
	@echo "══════════════════════════════════════════════════════════════"

