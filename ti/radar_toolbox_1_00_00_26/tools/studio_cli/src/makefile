###################################################################################
# mmWave Demo Level makefile
###################################################################################
ifeq ($(CCS_MAKEFILE_BASED_BUILD), 1)
ifeq ($(OS),Windows_NT)
include $(MMWAVE_SDK_INSTALL_PATH)/scripts/windows/setenv.mak
else
include $(MMWAVE_SDK_INSTALL_PATH)/scripts/unix/setenv.mak
endif
endif


# Set RadarSS firmware path based on device
ifeq ($(MMWAVE_SDK_DEVICE), awr68xx) 
RADAR_FIRMWARE_PATH  = $(XWR68XX_RADARSS_IMAGE_BIN)
MMWAVE_SDK_SHMEM_ALLOC=0x00000006
endif
ifeq ($(MMWAVE_SDK_DEVICE), iwr68xx)
RADAR_FIRMWARE_PATH  = $(XWR68XX_RADARSS_IMAGE_BIN)
MMWAVE_SDK_SHMEM_ALLOC=0x00000006
endif
ifeq ($(MMWAVE_SDK_DEVICE), iwr18xx)
RADAR_FIRMWARE_PATH  = $(XWR18XX_RADARSS_IMAGE_BIN)
MMWAVE_SDK_SHMEM_ALLOC=0x00000008
endif
ifeq ($(MMWAVE_SDK_DEVICE), awr18xx)
RADAR_FIRMWARE_PATH  = $(XWR18XX_RADARSS_IMAGE_BIN)
MMWAVE_SDK_SHMEM_ALLOC=0x00000008
endif
ifeq ($(MMWAVE_SDK_DEVICE), iwr16xx)
RADAR_FIRMWARE_PATH  = $(XWR16XX_RADARSS_IMAGE_BIN)
MMWAVE_SDK_SHMEM_ALLOC=0x00000006
endif
ifeq ($(MMWAVE_SDK_DEVICE), awr16xx)
RADAR_FIRMWARE_PATH  = $(XWR16XX_RADARSS_IMAGE_BIN)
MMWAVE_SDK_SHMEM_ALLOC=0x00000006
endif
ifeq ($(MMWAVE_SDK_DEVICE), iwr14xx)
RADAR_FIRMWARE_PATH  = $(XWR14XX_RADARSS_IMAGE_BIN)
MMWAVE_SDK_SHMEM_ALLOC= 0x00010005
endif
ifeq ($(MMWAVE_SDK_DEVICE), awr14xx)
RADAR_FIRMWARE_PATH  = $(XWR14XX_RADARSS_IMAGE_BIN)
MMWAVE_SDK_SHMEM_ALLOC= 0x00010005
endif

include $(MMWAVE_SDK_INSTALL_PATH)/ti/common/mmwave_sdk.mak
include ./mss/mmw_mss.mak

MMW_DEMO_BIN         = mmwave_Studio_cli_$(MMWAVE_SDK_DEVICE_TYPE).bin
MMW_DEMO_SECURE_BIN  = mmwave_Studio_cli_$(MMWAVE_SDK_DEVICE_TYPE)_secure.bin

###################################################################################
# Standard Targets which need to be implemented by each mmWave SDK module. This
# plugs into the release scripts.
###################################################################################
.PHONY: all clean
.NOTPARALLEL:

# Clean: This cleans all the objects
clean: mssDemoClean 
	@$(DEL) $(MMW_DEMO_BIN)
	@$(DEL) $(MMW_DEMO_SECURE_BIN)

# This builds all the supported demos
all: mssDemo 
	@echo '**#########################################*'
	@$(GENERATE_METAIMAGE) $(MMW_DEMO_BIN) $(SHMEM_ALLOC) $(MSS_MMW_DEMO_OUT)   $(RADAR_FIRMWARE_PATH)  NULL
	@$(GENERATE_HS_METAIMAGE) $(MMW_DEMO_SECURE_BIN) $(SHMEM_ALLOC) $(MSS_MMW_DEMO_OUT) $(RADAR_FIRMWARE_PATH) $(MMWAVE_SECDEV_HSIMAGE_CFG)

# Help: This displays the MAKEFILE Usage.
help:
	@echo '****************************************************************************************'
	@echo '* Makefile Targets for the DEMO '
	@echo 'all	       -> Builds the mmw Demo (both MSS and DSS binaries)'
	@echo 'clean           -> Cleans the mmw Demo (both MSS and DSS binaries)'
	@echo 'mssDemo         -> Builds the MSS binary for the mmw Demo'
	@echo 'mssDemoClean    -> Cleans the MSS binary for mmw Demo '
	@echo '****************************************************************************************'

