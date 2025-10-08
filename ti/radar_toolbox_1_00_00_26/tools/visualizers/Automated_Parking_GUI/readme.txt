MATLAB GUI EXECUTION

mmw_demo_dpc ('xwr18xx',Data Port COM (Standard COM),Range_m,Width_m,User UART COM (Enhanced COM Port),'profile_name.cfg',1)

Specify arguments in this order: <platform> <comportSnum> <range_depth> <range_width> <comportCliNum> <cliCfgFileName> <loadCfg> 
platform                : 'xwr18xx' 
comportSnum             : comport on which visualization data is being sent (DATA Uart)
range_depth (in meters) : determines the y-axis of the plot 
range_width (in meters) : determines the x-axis (one sided) of the plot 
comportCliNum           : comport over which the cli configuration is sent to sensor (USER Uart)
cliCfgFileName          : Input cli configuration file 
loadCfg                 : loadCfg=1: cli configuration is sent to sensor 
                          loadCfg=0: it is assumed that configuration is already sent to sensor
                                     and it is already running, the configuration is not sent to sensor, 
                                     but it will keep receiving and displaying incomming data.

Examples:
AOP

     From DOS cmd
     ------------
         mmw_demo_parking_xyz.exe xwr1843 13 10 4 12 ..\chirp_profiles\profile_parking_mimo_2d_50m_3d_10m_aop.cfg 1
	 
	 
     From Matlab cmd
	 ---------------
         mmw_demo_parking_xyz ('xwr18xx',13,10,12,'..\chirp_profiles\profile_parking_mimo_2d_50m_3d_10m_aop.cfg',1)
	 
	 
BOOST

     From DOS cmd
     ------------
        mmw_demo_parking_xyz.exe xwr1843 35 10 4 36 ..\chirp_profiles\profile_parking_mimo_2d_50m_3d_10m_boost.cfg 1
	 
	 
     From Matlab cmd
	 ---------------
       mmw_demo_parking_xyz ('xwr18xx',35,10,4,36,'..\chirp_profiles\profile_parking_mimo_2d_50m_3d_10m_boost.cfg',1)






MATLAB Compiler

1. Prerequisites for Deployment 

. Verify the MATLAB Runtime is installed and ensure you    
  have installed version 9.2 (R2017a).   

. If the MATLAB Runtime is not installed, do the following:
  (1) enter
  
      >>mcrinstaller
      
      at MATLAB prompt. The MCRINSTALLER command displays the 
      location of the MATLAB Runtime installer.

  (2) run the MATLAB Runtime installer.

Or download the Windows 64-bit version of the MATLAB Runtime for R2017a 
from the MathWorks Web site by navigating to

   http://www.mathworks.com/products/compiler/mcr/index.html
   
   
For more information about the MATLAB Runtime and the MATLAB Runtime installer, see 
Package and Distribute in the MATLAB Compiler documentation  
in the MathWorks Documentation Center.    


NOTE: You will need administrator rights to run MCRInstaller. 


2. Files to Deploy and Package

Files to package for Standalone 
================================
-mmw_demo_parking_xy.exe
-MCRInstaller.exe 
   -if end users are unable to download the MATLAB Runtime using the above  
    link, include it when building your component by clicking 
    the "Runtime downloaded from web" link in the Deployment Tool
-This readme file 

3. Definitions

For information on deployment terminology, go to 
http://www.mathworks.com/help. Select MATLAB Compiler >   
Getting Started > About Application Deployment > 
Deployment Product Terms in the MathWorks Documentation 
Center.





