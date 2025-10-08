##################################################################################
# GTRACK makefile
##################################################################################
include $(MMWAVE_SDK_INSTALL_PATH)/ti/common/mmwave_sdk.mak
include ./gtracklib.mak


.PHONY: all clean
.PHONY: gtracklib3d gtracklib3dClean

##################################################################################
# Build/Clean the driver
##################################################################################

# This builds the 3D GTRACK Library
gtracklib3d:
	$(MAKE) -s MMWAVE_SDK_LIB_BUILD_OPTION=3D gtracklib

# This cleans the 3D GTRACK Library
gtracklib3dClean:
	$(MAKE) -s MMWAVE_SDK_LIB_BUILD_OPTION=3D gtracklibClean

# Clean: This cleans all the objects
clean: gtracklib3dClean

# Build everything
all: gtracklib3d

# Help: This displays the MAKEFILE Usage.
help:
	@echo '****************************************************************************************'
	@echo '* Makefile Targets for GTRACK '
	@echo 'all            -> Build gtrack lib'
	@echo 'clean          -> Clean out all the gtrack objects'
	@echo '****************************************************************************************'
