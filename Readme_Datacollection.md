# Data collection readme

I recommend using VS Code. Open this repo in VS code.

## Setting up the board

1. Connect power, usb and ethernet cable to radar (red board).
2. Connect usb to RADAR_FTDI J1 (port name written on the board) port on DCA1000EVM (gred board). There are two usb ports on this board, make sure the port is correct.

## CLI data collection

Following procedure is based on ```ti/radar_toolbox_1_00_00_26/tools/studio_cli/docs/mmwave_studio_CLI_Getting_Started_Guide.html``` with some additions.

1. Configure your laptop's ethernet properties same as step 3 from: https://www.ti.com/lit/ml/spruik7/spruik7.pdf?ts=1759882289366
2. Install VSCode_CLI_command_utility from ```ti\radar_toolbox_1_00_00_26\VSCode_CLI_command_utility```. How to install: https://www.youtube.com/watch?v=Z724l3mq2ag&t=29s. 
3. Install Matlab runtime engine (v8.5.1 32-bit). This specific version only.
4. mmWave radar configurations can be tweaked from ```ti\radar_toolbox_1_00_00_26\tools\studio_cli\src\profiles\profile_monitor_xwr18xx.cfg```. Make changes only if required. VSCode utility will help understanding what these setting mean.
5. Open terminal and cd to ```ti\radar_toolbox_1_00_00_26\tools\studio_cli\gui\mmw_cli_tool\```.
6. Run ```./mmwave_studio_cli.exe```. It will configure the radar.
7. When the script is done configuring the radar, run the following commands to start chirps and gather ADC output.
    - sensorStop
    - dcastart
    - sensorStart
    - sensorStop
    - dcastop


