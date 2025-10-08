###################################################################################
# Millimeter Wave Demo
###################################################################################
.PHONY: mssDemo mssDemoClean

###################################################################################
# Setup the VPATH:
###################################################################################
vpath %.c ./common \
          ./mss

###################################################################################
# Additional libraries which are required to build the DEMO:
###################################################################################
MSS_MMW_DEMO_STD_LIBS = $(R4F_COMMON_STD_LIB)									\
			-llibpinmux_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT) 		\
		   	-llibcrc_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)			\
		   	-llibuart_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)			\
		   	-llibmailbox_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)		\
		   	-llibmmwavelink_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)	\
		   	-llibadcbuf_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)      	\
		   	-llibdma_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)         	\
		   	-llibgpio_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)         	\
		   	-llibedma_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)			\
		   	-llibmathutils.$(R4F_LIB_EXT)                               \
            -llibosal_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)          \
            -llibcbuff_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)         \
            -llibhsiheader_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)

MSS_MMW_DEMO_LOC_LIBS = $(R4F_COMMON_LOC_LIB)									\
   			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/pinmux/lib 	   	\
   			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/uart/lib		\
   			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/crc/lib			\
   			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/mailbox/lib	    \
		   	-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/adcbuf/lib	    \
   			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/dma/lib         \
   			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/gpio/lib        \
   			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/edma/lib		\
   			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/control/mmwavelink/lib	\
   			-i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/mathutils/lib     \
            -i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/osal/lib        \
            -i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/cbuff/lib       \
            -i$(MMWAVE_SDK_INSTALL_PATH)/ti/utils/hsiheader/lib


###################################################################################
# Millimeter Wave Demo
###################################################################################
MSS_MMW_CFG_PREFIX       = mmw_mss
MSS_MMW_DEMO_CFG         = $(MSS_MMW_CFG_PREFIX).cfg
MSS_MMW_DEMO_ROV_XS      = $(MSS_MMW_CFG_PREFIX)_$(R4F_XS_SUFFIX).rov.xs
MSS_MMW_DEMO_CONFIGPKG   = mmw_configPkg_mss_$(MMWAVE_SDK_DEVICE_TYPE)
MSS_MMW_DEMO_MAP         = mmwave_Studio_cli_$(MMWAVE_SDK_DEVICE_TYPE)_mss.map
MSS_MMW_DEMO_OUT         = mmwave_Studio_cli_$(MMWAVE_SDK_DEVICE_TYPE)_mss.$(R4F_EXE_EXT)
MSS_MMW_DEMO_METAIMG_BIN = mmwave_Studio_cli_$(MMWAVE_SDK_DEVICE_TYPE).bin
MSS_MMW_DEMO_CMD         = mss/mmw_mss_linker.cmd
MSS_MMW_DEMO_SOURCES     =  mss_main.c \
                       mmw_cli.c \
                       mmw_lvds_stream.c \
					   cli_rfmonitor.c \
					   rfmonitor.c \
					   rfmonitor_configdefaults.c \
                       mmw_rfparser.c \
                       mmw_adcconfig.c \
                       osi_tirtos.c \
                       mmwl_if.c


MSS_MMW_DEMO_DEPENDS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_DEMO_SOURCES:.c=.$(R4F_DEP_EXT)))
MSS_MMW_DEMO_OBJECTS   = $(addprefix $(PLATFORM_OBJDIR)/, $(MSS_MMW_DEMO_SOURCES:.c=.$(R4F_OBJ_EXT)))

R4F_CFLAGS  = -mv7R4 --code_state=16 --float_support=VFPv3D16 --abi=eabi -me            \
              --define=SUBSYS_MSS --define=$(PLATFORM_DEFINE)     \
              --define=_LITTLE_ENDIAN --define=DebugP_ASSERT_ENABLED $(R4F_INCLUDE) 	\
              -g -display_error_number --diag_warning=225 --diag_wrap=off 			\
              --little_endian --preproc_with_compile --gen_func_subsections 			\
			  $(R4F_CFLAGS_ENUM_TYPE)

###################################################################################
# RTSC Configuration:
###################################################################################
mmwMssRTSC:
	@echo 'Configuring RTSC packages...'
	$(XS) --xdcpath="$(XDCPATH)" xdc.tools.configuro $(R4F_XSFLAGS) -o $(MSS_MMW_DEMO_CONFIGPKG) mss/$(MSS_MMW_DEMO_CFG)
	@echo 'Finished configuring packages'
	@echo ' '

###################################################################################
# Build the Millimeter Wave Demo
###################################################################################
mssDemo: BUILD_CONFIGPKG=$(MSS_MMW_DEMO_CONFIGPKG)
mssDemo: R4F_CFLAGS += --cmd_file=$(BUILD_CONFIGPKG)/compiler.opt \
                       --define=LIMITED_FEATURES \
                       --define=DebugP_LOG_ENABLED \
                       --define=SUBSYS_MSS
mssDemo: buildDirectories mmwMssRTSC $(MSS_MMW_DEMO_OBJECTS)
	$(R4F_LD) $(R4F_LDFLAGS) $(MSS_MMW_DEMO_LOC_LIBS) $(MSS_MMW_DEMO_STD_LIBS) 					\
	-l$(MSS_MMW_DEMO_CONFIGPKG)/linker.cmd --map_file=$(MSS_MMW_DEMO_MAP) $(MSS_MMW_DEMO_OBJECTS) 	\
	$(PLATFORM_R4F_LINK_CMD) $(MSS_MMW_DEMO_CMD) $(R4F_LD_RTS_FLAGS) -o $(MSS_MMW_DEMO_OUT)
	$(COPY_CMD) $(MSS_MMW_DEMO_CONFIGPKG)/package/cfg/$(MSS_MMW_DEMO_ROV_XS) $(MSS_MMW_DEMO_ROV_XS)
	@echo '******************************************************************************'
	@echo 'Built the MSS for Millimeter Wave Demo'
	@echo '******************************************************************************'

###################################################################################
# Cleanup the Millimeter Wave Demo
###################################################################################
mssDemoClean:
	@echo 'Cleaning the Millimeter Wave Demo MSS Objects'
	@rm -f $(MSS_MMW_DEMO_OBJECTS) $(MSS_MMW_DEMO_MAP) $(MSS_MMW_DEMO_OUT) $(MSS_MMW_DEMO_METAIMG_BIN) $(MSS_MMW_DEMO_DEPENDS) $(MSS_MMW_DEMO_ROV_XS)
	@echo 'Cleaning the Millimeter Wave Demo MSS RTSC package'
	@$(DEL) $(MSS_MMW_DEMO_CONFIGPKG)
	@$(DEL) $(PLATFORM_OBJDIR)

###################################################################################
# Dependency handling
###################################################################################
-include $(MSS_MMW_DEMO_DEPENDS)

