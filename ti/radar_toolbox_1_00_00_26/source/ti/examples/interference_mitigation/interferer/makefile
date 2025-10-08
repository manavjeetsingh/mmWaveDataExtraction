###################################################################################
# Interferer MSS Makefile
###################################################################################


include $(MMWAVE_SDK_INSTALL_PATH)/ti/common/mmwave_sdk.mak

.PHONY: all clean

###################################################################################
# Include libraries
###################################################################################
INTERFERER_STD_LIBS = $(R4F_COMMON_STD_LIB)									\
		   						-llibpinmux_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT) 		\
		   						-llibcrc_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)			\
		   						-llibmailbox_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)		\
		   						-llibadcbuf_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)    \
		   						-llibmmwavelink_$(MMWAVE_SDK_DEVICE_TYPE).$(R4F_LIB_EXT)
INTERFERER_LOC_LIBS = $(R4F_COMMON_LOC_LIB) 									\
		   						-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/pinmux/lib 		\
		   						-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/adcbuf/lib	   \
		   						-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/crc/lib			\
		   						-i$(MMWAVE_SDK_INSTALL_PATH)/ti/drivers/mailbox/lib		\
		   						-i$(MMWAVE_SDK_INSTALL_PATH)/ti/control/mmwavelink/lib

###################################################################################
# Interferer Files
###################################################################################
INTERFERER_CFG		 = mss.cfg
INTERFERER_CMD		 = ti/platform/$(MMWAVE_SDK_DEVICE_TYPE)
INTERFERER_CONFIGPKG	 = mss_configPkg_$(MMWAVE_SDK_DEVICE_TYPE)
INTERFERER_MAP		 = $(MMWAVE_SDK_DEVICE_TYPE)_interferer_mss.map
INTERFERER_OUT		 = $(MMWAVE_SDK_DEVICE_TYPE)_interferer_mss.$(R4F_EXE_EXT)
INTERFERER_BIN		 = $(MMWAVE_SDK_DEVICE_TYPE)_interferer.bin
INTERFERER_APP_CMD	 = mss_interferer_linker.cmd
INTERFERER_SOURCES	 = main_mss.c osi_tirtos.c
INTERFERER_DEPENDS	 = $(addprefix $(PLATFORM_OBJDIR)/, $(INTERFERER_SOURCES:.c=.$(R4F_DEP_EXT)))
INTERFERER_OBJECTS	 = $(addprefix $(PLATFORM_OBJDIR)/, $(INTERFERER_SOURCES:.c=.$(R4F_OBJ_EXT)))

ifeq ($(MMWAVE_SDK_DEVICE_TYPE), xwr16xx)
INTERFERER_RADARSS_IMAGE_BIN = $(XWR16XX_RADARSS_IMAGE_BIN)
endif
ifeq ($(MMWAVE_SDK_DEVICE_TYPE), xwr18xx)
INTERFERER_RADARSS_IMAGE_BIN = $(XWR18XX_RADARSS_IMAGE_BIN)
endif

###################################################################################
# RTSC Configuration:
###################################################################################
linkRTSC: $(R4_CFG)
	@echo 'Configuring RTSC packages...'
	$(XS) --xdcpath="$(XDCPATH)" xdc.tools.configuro $(R4F_XSFLAGS) -o $(INTERFERER_CONFIGPKG) $(INTERFERER_CFG)
	@echo 'Finished configuring packages'
	@echo ' '

###################################################################################
# Build interferer:
###################################################################################
all: BUILD_CONFIGPKG=$(INTERFERER_CONFIGPKG)
all: R4F_CFLAGS += --cmd_file=$(BUILD_CONFIGPKG)/compiler.opt
all: buildDirectories linkRTSC $(INTERFERER_OBJECTS)
	$(R4F_LD) $(R4F_LDFLAGS) $(INTERFERER_LOC_LIBS) $(INTERFERER_STD_LIBS) 	\
	-l$(INTERFERER_CONFIGPKG)/linker.cmd --map_file=$(INTERFERER_MAP) 		\
	$(INTERFERER_OBJECTS) $(PLATFORM_R4F_LINK_CMD) $(INTERFERER_APP_CMD) 	\
	$(R4F_LD_RTS_FLAGS) -o $(INTERFERER_OUT)
	@$(GENERATE_METAIMAGE) $(INTERFERER_BIN) $(SHMEM_ALLOC) $(INTERFERER_OUT) $(INTERFERER_RADARSS_IMAGE_BIN) NULL
	@echo '******************************************************************************'
	@echo 'Built the interferer '
	@echo '******************************************************************************'

###################################################################################
# Cleanup interferer:
###################################################################################
clean:
	@echo 'Cleaning interferer objects'
	@$(DEL) $(INTERFERER_OBJECTS) $(INTERFERER_OUT) $(INTERFERER_BIN)
	@$(DEL) $(INTERFERER_MAP) $(INTERFERER_DEPENDS)
	@echo 'Cleaning the Interferer MSS Unit RTSC package'
	@$(DEL) $(INTERFERER_CONFIGPKG)
	@$(DEL) $(PLATFORM_OBJDIR)

###################################################################################
# Dependency handling
###################################################################################
-include $(INTERFERER_DEPENDS)

