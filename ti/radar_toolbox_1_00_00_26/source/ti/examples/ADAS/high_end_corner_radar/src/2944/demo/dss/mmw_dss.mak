###################################################################################
# Millimeter Wave Demo
###################################################################################

.PHONY: dssDemo dssDemoClean 
#mmwDssRTSC
###################################################################################
# Setup the VPATH:
###################################################################################
vpath %.c ./utils \
          ../datapath/dpc/objectdetection/objdethwaDDMA/src \
		  ./dss

DSS_CPU := C66
DSS_CPU_INSTANCE := c66
DSS_CPU_INSTANCE_NUM := c66xdsp_1

###################################################################################
# Additional libraries which are required to build the DEMO:
###################################################################################
DSS_MMW_DEMO_STD_LIBS = $($(DSS_CPU)_COMMON_STD_LIB)						\
			-llibdpm_$(MMWAVE_SDK_DEVICE_TYPE).$($(DSS_CPU)_LIB_EXT) 		\
			-llibmathutils.$($(DSS_CPU)_LIB_EXT) 					\
			-llibrangeproc_hwa_ddma_$(MMWAVE_SDK_DEVICE_TYPE).$($(DSS_CPU)_LIB_EXT)     \
			-llibdopplerproc_hwa_ddma_$(MMWAVE_SDK_DEVICE_TYPE).$($(DSS_CPU)_LIB_EXT)   \
			-llibrangecfarproc_hwa_ddma_$(MMWAVE_SDK_DEVICE_TYPE).$($(DSS_CPU)_LIB_EXT)   \
			-llibdpedma_hwa_$(MMWAVE_SDK_DEVICE_TYPE).$($(DSS_CPU)_LIB_EXT) \
		    -lmathlib.$($(DSS_CPU)_LIB_EXT) \
			-ldsplib.$($(DSS_CPU)_LIB_EXT) 					

DSS_MMW_DEMO_LOC_LIBS = $($(DSS_CPU)_COMMON_LOC_LIB)						\
			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/control/dpm/lib         	\
			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/mathutils/lib     	\
			-i$($(DSS_CPU)x_MATHLIB_INSTALL_PATH)/packages/ti/mathlib/lib \
			-i../datapath/dpu/rangeprocDDMA/lib          \
			-i../datapath/dpc/dpu/dopplerprocDDMA/lib    \
			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpc/dpu/rangecfarprocDDMA/lib    \
			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/datapath/dpedma/lib                 \
			-i$($(DSS_CPU)x_DSPLIB_INSTALL_PATH)/packages/ti/dsplib/lib 

###################################################################################
# Millimeter Wave Demo
###################################################################################
DSS_MMW_CFG_PREFIX       = mmw_dss
DSS_MMW_DEMO_MAP         = awr2944_corner_radar_dss.map
DSS_MMW_DEMO_OUT         = awr2944_corner_radar_dss.$($(DSS_CPU)_EXE_EXT)
DSS_MMW_DEMO_CMD         = dss/mmw_dss_linker.cmd
DSS_MMW_DEMO_SOURCES     =  dss_main.c \
							data_path.c \
							objectdetection.c 

DSS_MMW_DEMO_SOURCES_GEN  = 	 ti_board_config.c	\
								 ti_board_open_close.c	\
								 ti_dpl_config.c	\
								 ti_drivers_config.c	\
								 ti_pinmux_config.c	\
								 ti_power_clock_config.c	\
								 ti_drivers_open_close.c
								 
DSS_MMW_DEMO_OBJECTS_GEN = $(addprefix $(PLATFORM_OBJDIR)/dssgenerated/, $(DSS_MMW_DEMO_SOURCES_GEN:.c=.$($(DSS_CPU)_OBJ_EXT)))
DSS_MMW_DEMO_DEPENDS   = $(addprefix $(PLATFORM_OBJDIR)/, $(DSS_MMW_DEMO_SOURCES:.c=.$($(DSS_CPU)_DEP_EXT)))
DSS_MMW_DEMO_OBJECTS   = $(addprefix $(PLATFORM_OBJDIR)/, $(DSS_MMW_DEMO_SOURCES:.c=.$($(DSS_CPU)_OBJ_EXT)))

###################################################################################
# Build the Millimeter Wave Demo
###################################################################################
dssDemo: $(DSS_CPU)_CFLAGS += -i$($(DSS_CPU)x_MATHLIB_INSTALL_PATH)/packages \
                        --define=DebugP_LOG_ENABLED

dssDemo: buildDirectories dssbuildDirectories $(DSS_MMW_DEMO_OBJECTS) $(DSS_MMW_DEMO_OBJECTS_GEN)
	$($(DSS_CPU)_LD) $($(DSS_CPU)_LDFLAGS) $(DSS_MMW_DEMO_LOC_LIBS) $(DSS_MMW_DEMO_STD_LIBS) 	\
	--map_file=$(DSS_MMW_DEMO_MAP) $(DSS_MMW_DEMO_OBJECTS) 	$(DSS_MMW_DEMO_OBJECTS_GEN) \
	$(PLATFORM_$(DSS_CPU)X_LINK_CMD) $(DSS_MMW_DEMO_CMD) -o $(DSS_MMW_DEMO_OUT)
	@echo '******************************************************************************'
	@echo 'Built the DSS for Millimeter Wave Demo (DDMA)'
	@echo '******************************************************************************'


###################################################################################
# Cleanup the Millimeter Wave Demo
###################################################################################
dssDemoClean:
	@echo 'Cleaning the Millimeter Wave Demo DSS Objects'
	@rm -f $(DSS_MMW_DEMO_OBJECTS) $(DSS_MMW_DEMO_MAP) $(DSS_MMW_DEMO_OUT) $(DSS_MMW_DEMO_DEPENDS) $(DSS_MMW_DEMO_OBJECTS_GEN)
#	@echo 'Cleaning the Millimeter Wave Demo DSS RTSC package'
	@$(DEL) $(PLATFORM_OBJDIR)

###################################################################################
# Dependency handling
###################################################################################
-include $(DSS_MMW_DEMO_DEPENDS)

