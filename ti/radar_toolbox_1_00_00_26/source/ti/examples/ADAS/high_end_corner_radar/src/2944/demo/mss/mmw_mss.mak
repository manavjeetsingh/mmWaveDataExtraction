###################################################################################
# Millimeter Wave Demo
###################################################################################
.PHONY: mssDemo mssDemoClean
###################################################################################
# Setup the VPATH:
###################################################################################
vpath %.c ./utils \
          ./mss

MSS_CPU := R5F
MSS_CPU_INSTANCE := r5f

###################################################################################
# Additional libraries which are required to build the DEMO:
###################################################################################
MSS_MMW_DEMO_STD_LIBS = $($(MSS_CPU)_COMMON_STD_LIB) \
						-llibtestlogger_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT)		\
						-lmmwavelink_r5f.lib	\
						-llibmmwave_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT) \
						-llibdpm_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT) \
						-llibmathutils.$($(MSS_CPU)_LIB_EXT) \
						-llibcli_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT)\
						-llibgtrack3D_$(MMWAVE_SDK_DEVICE_TYPE).$($(MSS_CPU)_LIB_EXT)


MSS_MMW_DEMO_LOC_LIBS = $($(MSS_CPU)_COMMON_LOC_LIB)  \
						-Wl,-i$(MMWAVE_AWR294X_DFP_INSTALL_PATH)/ti/control/mmwavelink/lib \
						-Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/control/mmwave/lib \
						-Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/cli/lib \
						-Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/control/dpm/lib \
						-Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/mathutils/lib \
						-Wl,-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/testlogger/lib \
						-Wl,-i../alg/gtrack/lib/

###################################################################################
# Millimeter Wave Demo
###################################################################################
MSS_MMW_DEMO_MAP         = awr2944_corner_radar_mss.map
MSS_MMW_DEMO_OUT         = awr2944_corner_radar_mss.$($(MSS_CPU)_EXE_EXT)
MSS_MMW_DEMO_CMD         = mss/mmw_mss_linker.cmd
MSS_MMW_DEMO_SOURCES     = mmwdemo_rfparserDDMA.c  \
                           mmwdemo_adcconfig.c \
                           mmw_cli.c \
                           mss_main.c \
						   gtrackAlloc.c \
						   gtrackLog.c

MSS_MMW_DEMO_SOURCES_GEN  = ti_board_config.c	\
                            ti_board_open_close.c	\
                            ti_dpl_config.c	\
                            ti_drivers_config.c	\
                            ti_pinmux_config.c	\
                            ti_power_clock_config.c	\
                            ti_drivers_open_close.c


MSS_MMW_DEMO_DEPENDS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_DEMO_SOURCES:.c=.$($(MSS_CPU)_DEP_EXT)))
MSS_MMW_DEMO_OBJECTS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_DEMO_SOURCES:.c=.$($(MSS_CPU)_OBJ_EXT)))
MSS_MMW_DEMO_OBJECTS_GEN = $(addprefix $(PLATFORM_OBJDIR)/mssgenerated/, $(MSS_MMW_DEMO_SOURCES_GEN:.c=.$($(MSS_CPU)_OBJ_EXT)))
                
###################################################################################
# Build the Millimeter Wave Demo
###################################################################################
mssDemo: $(MSS_CPU)_CFLAGS += -DDebugP_LOG_ENABLED
mssDemo: buildDirectories mssbuildDirectories $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_DEMO_OBJECTS_GEN)
	$($(MSS_CPU)_LD) $($(MSS_CPU)_LDFLAGS) $(MSS_MMW_DEMO_LOC_LIBS) -Wl,-m=$(MSS_MMW_DEMO_MAP) \
	-o $(MSS_MMW_DEMO_OUT) $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_DEMO_OBJECTS_GEN) $(MSS_MMW_DEMO_STD_LIBS) \
	$(PLATFORM_$(MSS_CPU)_LINK_CMD) $(MSS_MMW_DEMO_CMD)
	@echo '******************************************************************************'
	@echo 'Built the MSS for Millimeter Wave Demo'
	@echo '******************************************************************************'

###################################################################################
# Cleanup the Millimeter Wave Demo
###################################################################################
mssDemoClean:
	@echo 'Cleaning the Millimeter Wave Demo MSS Objects'
	@rm -f $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_DEMO_OBJECTS_GEN) $(MSS_MMW_DEMO_MAP) 
	@rm -f $(MSS_MMW_DEMO_OUT) $(MSS_MMW_DEMO_DEPENDS) $(MSS_MMW_DEMO_ROV_XS)
	@$(DEL) $(PLATFORM_OBJDIR)


###################################################################################
# Dependency handling
###################################################################################
-include $(MSS_MMW_DEMO_DEPENDS)

