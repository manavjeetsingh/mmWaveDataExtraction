##################################################################################
# rangeproc makefile
##################################################################################
# The following needs to be before any makefile inclusion
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MODULE_NAME := $(notdir $(patsubst %/,%,$(dir $(MAKEFILE_PATH))))

include $(MMWAVE_SDK_INSTALL_PATH)/ti/common/mmwave_sdk.mak

##################################################################################
# SOC Specific Test Targets
##################################################################################
ifneq ($(filter $(MMWAVE_SDK_DEVICE_TYPE), awr2944), )
include ./rangeproclib.mak
# include ./test/mssHWATest.mak
# include ./test/dssHWATest.mak
# include ./test/dssHWADDMATest.mak

###################################################################################
# Standard Targets which need to be implemented by each mmWave SDK module. This
# plugs into the release scripts.
###################################################################################
.PHONY: all clean lib libClean help # test testClean 

##################################################################################
# Build/Clean the library 
##################################################################################

# This builds the lib
lib: rangeprocLib

# This cleans the lib
libClean: rangeprocLibClean

# Clean: This cleans all the objects
clean: libClean

# Build everything
all: lib #test

# Help: This displays the MAKEFILE Usage.
help:
	@echo '****************************************************************************************'
	@echo '* Makefile Targets for the Range Proc DPU'
	@echo 'all               -> Build lib and all unit tests'
	@echo 'clean             -> Clean out all the objects'
	@echo '****************************************************************************************'

else
.PHONY: lib

lib test .DEFAULT:
	@echo "Error: $(MODULE_NAME) is not supported on $(MMWAVE_SDK_DEVICE) !!!"
endif

