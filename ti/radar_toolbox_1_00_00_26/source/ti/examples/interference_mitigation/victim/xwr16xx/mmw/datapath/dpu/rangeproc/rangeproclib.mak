###################################################################################
# rangeproc Library Makefile
###################################################################################
.PHONY: rangeprocLib rangeprocLibClean

###################################################################################
# Setup the VPATH:
###################################################################################
vpath %.c src
vpath %.c platform

###################################################################################
# Library Source Files:
###################################################################################
RANGEPROC_DSP_LIB_SOURCES = rangeprocdsp.c rangeproc_interference.c 			

###################################################################################
# Library objects
#     Build for R4 and DSP
###################################################################################
RANGEPROC_DSP_C674_DRV_LIB_OBJECTS = $(addprefix $(PLATFORM_OBJDIR)/, $(RANGEPROC_DSP_LIB_SOURCES:.c=.$(C674_OBJ_EXT)))

###################################################################################
# Library Dependency:
###################################################################################
RANGEPROC_DSP_C674_DRV_DEPENDS = $(addprefix $(PLATFORM_OBJDIR)/, $(RANGEPROC_DSP_LIB_SOURCES:.c=.$(C674_DEP_EXT)))

###################################################################################
# Library Names:
###################################################################################
RANGEPROC_DSP_C674_DRV_LIB = lib/librangeproc_dsp_$(MMWAVE_SDK_DEVICE_TYPE).$(C674_LIB_EXT)

###################################################################################
# Library Build:
#     - Build the R4 & DSP Library
###################################################################################
rangeprocDSPLib: C674_CFLAGS += -i$(C674x_MATHLIB_INSTALL_PATH)/packages \
								-i$(C64Px_DSPLIB_INSTALL_PATH)/packages/ti/dsplib/src/DSP_fft16x16_imre/c64P
rangeprocDSPLib: buildDirectories $(RANGEPROC_DSP_C674_DRV_LIB_OBJECTS)
	if [ ! -d "lib" ]; then mkdir lib; fi
	echo "Archiving $@"
	$(C674_AR) $(C674_AR_OPTS) $(RANGEPROC_DSP_C674_DRV_LIB) $(RANGEPROC_DSP_C674_DRV_LIB_OBJECTS)

rangeprocLib: rangeprocDSPLib

###################################################################################
# Clean the Libraries
###################################################################################
rangeprocDSPLibClean:
	@echo 'Cleaning the rangeproc DSP Library Objects'
	@$(DEL) $(RANGEPROC_DSP_C674_DRV_LIB_OBJECTS) $(RANGEPROC_DSP_C674_DRV_LIB)
	@$(DEL) $(RANGEPROC_DSP_C674_DRV_DEPENDS)
	@$(DEL) $(PLATFORM_OBJDIR)

rangeprocLibClean: rangeprocDSPLibClean

###################################################################################
# Dependency handling
###################################################################################
-include $(RANGEPROC_DSP_C674_DRV_DEPENDS)

