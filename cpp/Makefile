# Host compiler global settings
CC = g++
CLCC = xocc

CC_FLAGS = --std=c++11 -O3 -Wno-deprecated-declarations -fopenmp

BITSTREAM_NAME = NaiveBayes
HOST_EXE = ${BITSTREAM_NAME}

PLATFORM = ${AWS_PLATFORM}

HOST_DIR = src
KERNEL_DIR = kernel_src
KERNEL_TYPE = cpp

# Host and Kernel sources
HOST_SRCS = $(wildcard $(HOST_DIR)/*/*.cpp) $(wildcard $(HOST_DIR)/*.cpp)
KERNEL_SRCS_CPP = $(wildcard $(KERNEL_DIR)/*.cpp)

HOST_OBJECTS := $(HOST_SRCS:.cpp=.o)
KERNEL_OBJECTS := $(KERNEL_SRCS_CPP:.cpp=.xo)

# Include Libraries
HOST_LFLAGS = -lcoral-api

# Connecting kernels to specific memory banks
BANKS = --sp Classifier_0_1.m_axi_gmem0:bank0 \
		--sp Classifier_0_1.m_axi_gmem1:bank0 \
		--sp Classifier_0_1.m_axi_gmem2:bank0 \
		--sp Classifier_0_1.m_axi_gmem3:bank0 \
		--sp Classifier_0_1.m_axi_gmem4:bank0 \
		--sp Classifier_1_1.m_axi_gmem0:bank1 \
		--sp Classifier_1_1.m_axi_gmem1:bank1 \
		--sp Classifier_1_1.m_axi_gmem2:bank1 \
		--sp Classifier_1_1.m_axi_gmem3:bank1 \
		--sp Classifier_1_1.m_axi_gmem4:bank1 \
		--sp Classifier_2_1.m_axi_gmem0:bank2 \
		--sp Classifier_2_1.m_axi_gmem1:bank2 \
		--sp Classifier_2_1.m_axi_gmem2:bank2 \
		--sp Classifier_2_1.m_axi_gmem3:bank2 \
		--sp Classifier_2_1.m_axi_gmem4:bank2 \
		--sp Classifier_3_1.m_axi_gmem0:bank3 \
		--sp Classifier_3_1.m_axi_gmem1:bank3 \
		--sp Classifier_3_1.m_axi_gmem2:bank3 \
		--sp Classifier_3_1.m_axi_gmem3:bank3 \
		--sp Classifier_3_1.m_axi_gmem4:bank3

# Additional Vivado options
VIVADO_OPTS = --xp misc:enableGlobalHoldIter="True" \
		--xp vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=NoTimingRelaxation

all: host xbin

host: ${HOST_EXE}

check_platform_defined:
	$(if $(value AWS_PLATFORM),,$(error AWS_PLATFORM is not set))

# Building host
${HOST_EXE}: ${HOST_OBJECTS}
	${CC} ${CC_FLAGS} ${HOST_OBJECTS} ${HOST_LFLAGS} -o $@
	${RM} -rf ${HOST_OBJECTS}

xbin: check_platform_defined ${KERNEL_OBJECTS}
	${CLCC} -t hw --link -s --platform ${PLATFORM} ${BANKS} ${VIVADO_OPTS} ${KERNEL_OBJECTS} -o ${BITSTREAM_NAME}.xclbin
	${RM} -rf ${KERNEL_OBJECTS}

%.o: %.cpp
	${CC} ${CC_FLAGS} -c $< -o $@

# Building kernel
%.xo: %.cpp
	${CLCC} ${TARGET} --save-temps --platform ${PLATFORM} --kernel $(notdir $(basename $<)) -c $< -o $@

clean:
	${RM} -rf ${HOST_EXE} ${KERNEL_OBJECTS} ${HOST_OBJECTS} *.log *.dir *.xml *.dcp *.dat _sds iprepo *.tcl xilinx_aws-vu9p-f1_dynamic_5_0.hpfm .Xil sdaccel_* _x top_sp.ltx

cleanall: clean
	${RM} -rf ${BITSTREAM_NAME}*

help:
	@echo "Compile host executable"
	@echo "make host"
	@echo ""
	@echo "Compile host executable for CPU version"
	@echo "make"
	@echo ""
	@echo "Compile .xclbin file for system run"
	@echo "make xbin"
	@echo ""
	@echo "Clean working diretory"
	@echo "make clean"
	@echo "Clean working diretory and bitstream files"
	@echo "make cleanall"
