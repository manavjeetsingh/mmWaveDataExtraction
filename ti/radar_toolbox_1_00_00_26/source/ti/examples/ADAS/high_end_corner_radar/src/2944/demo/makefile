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

PROC_CHAIN := DDMA

include $(MMWAVE_SDK_INSTALL_PATH)/ti/common/mmwave_sdk.mak
include ./mss/mmw_mss.mak
include ./dss/mmw_dss.mak

MMW_DEMO_BIN         = awr2944_corner_radar.appimage


###################################################################################
# Standard Targets which need to be implemented by each mmWave SDK module. This
# plugs into the release scripts.
###################################################################################
.PHONY: all clean
.NOTPARALLEL:

mmwDemo: mssDemo dssDemo
	
bin:
	@$(GENERATE_METAIMAGE) $(MMW_DEMO_BIN) $(MSS_MMW_DEMO_OUT) $(AWR294X_RADARSS_IMAGE_BIN) $(DSS_MMW_DEMO_OUT)


binClean:
	@$(DEL) $(MMW_DEMO_BIN)

PACKAGE_TYPE = ETS

msssyscfg: ./mss/mss.syscfg
	@echo Generating SysConfig files ...
	$(SYSCFG_NODE) $(SYSCFG_CLI_PATH)/dist/cli.js --product $(SYSCFG_SDKPRODUCT) --context r5fss0-0 --part Default --package $(PACKAGE_TYPE) --output ./mss/mssgenerated/ ./mss/mss.syscfg

dsssyscfg: ./dss/dss.syscfg
	@echo Generating SysConfig files ...
	$(SYSCFG_NODE) $(SYSCFG_CLI_PATH)/dist/cli.js --product $(SYSCFG_SDKPRODUCT) --context c66ss0 --part Default --package $(PACKAGE_TYPE) --output ./dss/dssgenerated/ ./dss/dss.syscfg

# syscfg: This generates syscfg files
syscfg: msssyscfg dsssyscfg

# syscfg: This cleans syscfg files
syscfgclean:
	@echo 'Cleaning the syscfg files'
	@$(DEL) ./mss/mssgenerated
	@$(DEL) ./dss/dssgenerated	

# This builds all the supported demos but the binaries
all: syscfg mmwDemo bin

# Clean: This cleans all the objects
clean: mssDemoClean dssDemoClean binClean syscfgclean

# Help: This displays the MAKEFILE Usage.
help:
	@echo '****************************************************************************************'
	@echo '* Makefile Targets for the DEMO '
	@echo 'all             -> Builds the mmw Demo (both MSS and DSS binaries) and appimage'
	@echo 'clean           -> Cleans the mmw Demo (both MSS and DSS binaries) and appimage'
	@echo 'mssDemo         -> Builds the MSS binary for the mmw Demo'
	@echo 'mssDemoClean    -> Cleans the MSS binary for mmw Demo '
	@echo 'dssDemo         -> Builds the DSS binary for the mmw Demo'
	@echo 'dssDemoClean    -> Cleans the DSS binary for mmw Demo '
	@echo 'bin             -> Builds the appimage for the demo'
	@echo 'binClean        -> Cleans the appimage for the demo'
	@echo 'mmwDemo         -> Builds the mmw Demo (both MSS and DSS binaries) but not the appimage'
	@echo 'mmwDemoClean    -> Builds the mmw Demo (both MSS and DSS binaries)'
	@echo '****************************************************************************************'

