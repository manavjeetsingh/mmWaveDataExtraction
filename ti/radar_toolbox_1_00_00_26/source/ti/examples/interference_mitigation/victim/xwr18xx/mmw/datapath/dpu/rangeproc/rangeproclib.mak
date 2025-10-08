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
RANGEPROC_HWA_LIB_SOURCES = rangeprochwa.c rangeproc_interference.c 					

###################################################################################
# Library objects
#     Build for R4 and DSP
###################################################################################
RANGEPROC_HWA_C674_DRV_LIB_OBJECTS = $(addprefix $(PLATFORM_OBJDIR)/, $(RANGEPROC_HWA_LIB_SOURCES:.c=.$(C674_OBJ_EXT)))

###################################################################################
# Library Dependency:
###################################################################################
RANGEPROC_HWA_C674_DRV_DEPENDS = $(addprefix $(PLATFORM_OBJDIR)/, $(RANGEPROC_HWA_LIB_SOURCES:.c=.$(C674_DEP_EXT)))

###################################################################################
# Library Names:
###################################################################################
RANGEPROC_HWA_C674_DRV_LIB = lib/librangeproc_hwa_$(MMWAVE_SDK_DEVICE_TYPE).$(C674_LIB_EXT)

###################################################################################
# Library Build:
#     - Build the R4 & DSP Library
###################################################################################


rangeprocHWALib: buildDirectories $(RANGEPROC_HWA_R4F_DRV_LIB_OBJECTS) $(RANGEPROC_HWA_C674_DRV_LIB_OBJECTS)
	if [ ! -d "lib" ]; then mkdir lib; fi
	echo "Archiving $@"
	$(C674_AR) $(C674_AR_OPTS) $(RANGEPROC_HWA_C674_DRV_LIB) $(RANGEPROC_HWA_C674_DRV_LIB_OBJECTS)

rangeprocLib: rangeprocHWALib

###################################################################################
# Clean the Libraries
###################################################################################
rangeprocHWALibClean:
	@echo 'Cleaning the rangeproc HWA Library Objects'
	@$(DEL) $(RANGEPROC_HWA_C674_DRV_LIB_OBJECTS) $(RANGEPROC_HWA_C674_DRV_LIB)
	@$(DEL) $(RANGEPROC_HWA_C674_DRV_DEPENDS)
	@$(DEL) $(PLATFORM_OBJDIR)

rangeprocLibClean: rangeprocHWALibClean

###################################################################################
# Dependency handling
###################################################################################
-include $(RANGEPROC_HWA_C674_DRV_DEPENDS)

