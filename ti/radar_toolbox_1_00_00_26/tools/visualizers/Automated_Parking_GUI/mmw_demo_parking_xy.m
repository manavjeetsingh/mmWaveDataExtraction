
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %
 %      (C) Copyright 2019 Texas Instruments, Inc.
 %
 %  Redistribution and use in source and binary forms, with or without
 %  modification, are permitted provided that the following conditions
 %  are met:
 %
 %    Redistributions of source code must retain the above copyright
 %    notice, this list of conditions and the following disclaimer.
 %
 %    Redistributions in binary form must reproduce the above copyright
 %    notice, this list of conditions and the following disclaimer in the
 %    documentation and/or other materials provided with the
 %    distribution.
 %
 %    Neither the name of Texas Instruments Incorporated nor the names of
 %    its contributors may be used to endorse or promote products derived
 %    from this software without specific prior written permission.
 %
 %  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 %  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 %  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 %  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 %  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 %  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 %  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 %  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 %  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 %  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 %  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%platform    : supported platforms: xwr14xx or xwr16xx
%comportSnum : comport on which visualization data is being sent
%range_depth (in meters) : determines the y-axis of the plot
%range_width (in meters) : determines the x-axis (one sided) of the plot
%comportCliNum : comport over which the cli configuration is sent to XWR16xx
%cliCfgFileName : Input cli configuration file
%loadCfg : loadCfg=1: cli configuration is sent to XWR16xx, loadCfg=0: it is
%          assumed that configuration is already sent to XWR16xx, and it is
%          already running, the configuration is not sent to XWR16xx, but it
%          will keep receiving and displaying incomming data.
%maxRangeProfileYaxis : maximum range profile y-axis
%%%%%%%%%%%EXPECTED FORMAT OF DATA BEING RECEIVED%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Magic Number : [2 1 4 3  6 5  8 7] : 8 bytes
%Inter frame processing time in CPU cycles : 4 bytes 
%Noise Energy :  4 bytes (uint32)
%Number of valid objects : 4 bytes
%The following data are sent depending on the cli configuration
%Object Data :  10 bytes per object *MAX_NUM_OBJECTS = 500 bytes [see structure of object data below objOut_t]
%Range Profile : 1DfftSize * sizeof(uint16_t) bytes
%2D range bins at zero Doppler, complex symbols including all received
%virtual antennas : 1DfftSize * sizeof(uint32_t) * numVirtualAntennas
%Doppler-Range 2D FFT log magnitude matrix: 1DfftSize * 2Dfftsize * sizeof(uint16_t)
%
% typedef volatile struct objOut
% {
%     uint8_t   rangeIdx;
%     uint8_t   dopplerIdx ;
%     uint16_t  peakVal;
%     int16_t  x; //in meters
%     int16_t  y; //in meters
%     int16_t  z; //in meters
% } objOut_t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = mmw_demo_parking(platform, comportSnum, range_depth, range_width, comportCliNum, cliCfgFileName, loadCfg, maxRangeProfileYaxis, debugFlag)

fprintf('Starting UI for mmWave Demo ....\n'); 

if (nargin < 6)
    fprintf('!!ERROR:Missing arguments!!\n');
    fprintf('Specify arguments in this order: <platform> <comportSnum> <range_depth> <range_width> <comportCliNum> <cliCfgFileName> <loadCfg> \n');
    fprintf('platform                : ''xwr14xx'' or ''xwr16xx'' \n');
    fprintf('comportSnum             : comport on which visualization data is being sent \n');
    fprintf('range_depth (in meters) : determines the y-axis of the plot \n');
    fprintf('range_width (in meters) : determines the x-axis (one sided) of the plot \n');
    fprintf('comportCliNum           : comport over which the cli configuration is sent to XWR16xx \n');
    fprintf('cliCfgFileName          : Input cli configuration file \n');
    fprintf('loadCfg                 : loadCfg=1: cli configuration is sent to XWR16xx \n');
    fprintf('                          loadCfg=0: it is assumed that configuration is already sent to XWR16xx\n');
    fprintf('                                     and it is already running, the configuration is not sent to XWR16xx, \n');
    fprintf('                                     but it will keep receiving and displaying incomming data. \n');
    
    return;
end
if(ischar(comportSnum))
    comportSnum=str2num(comportSnum);
    range_depth=str2num(range_depth);
    range_width=str2num(range_width);
end

if (range_depth/4 <=2)
    range_grid_arc = .5;
elseif (range_depth <=2)
    range_grid_arc = .5;
elseif (range_depth <=4)
    range_grid_arc = 1;
elseif (range_depth <=20)
    range_grid_arc = 5;
elseif (range_depth <=24)
    range_grid_arc = 10;
else
    range_grid_arc = 15;
end

if exist('maxRangeProfileYaxis', 'var')
    if(ischar(maxRangeProfileYaxis))
        maxRangeProfileYaxis = str2num(maxRangeProfileYaxis);
    end
else
    maxRangeProfileYaxis = 1e6;
end


if exist('debugFlag', 'var')
    if strcmpi(debugFlag, 'debugFlag')
         debugFlag = 1;
    else
        debugFlag = 0;
    end
else
    debugFlag = 0;
end

global platformType
global gPlatform
if exist('platform', 'var')
    if strcmpi(platform, 'xwr16xx')
         platformType = hex2dec('a1642');
    elseif strcmpi(platform, 'xwr14xx')
        platformType = hex2dec('a1443');
    elseif strcmpi(platform, 'xwr18xx')
        platformType = hex2dec('a1842');
    elseif strcmpi(platform, 'xwr1843')
        platformType = hex2dec('a1843');
    elseif strcmpi(platform, 'xwr6843')
        platformType = hex2dec('a6843');
    elseif strcmpi(platform, 'xwr6843')
        platformType = hex2dec('a6843');
    elseif strcmpi(platform, 'xwr6843aop')
        platformType = hex2dec('a6843');
    else
        fprintf('Unknown platform \n');
        return
    end
else
    platformType = hex2dec('a1642');
end
gPlatform = platform;

if ischar(loadCfg)
    loadCfg = str2num(loadCfg);
end

global MAX_NUM_OBJECTS;
global OBJ_STRUCT_SIZE_BYTES ;
global SIDE_INFO_STRUCT_SIZE_BYTES ;
global TOTAL_PAYLOAD_SIZE_BYTES;

MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
MMWDEMO_UART_MSG_RANGE_PROFILE   = 2;
MMWDEMO_UART_MSG_NOISE_PROFILE   = 3;
MMWDEMO_UART_MSG_AZIMUT_STATIC_HEAT_MAP = 4;
MMWDEMO_UART_MSG_RANGE_DOPPLER_HEAT_MAP = 5;
MMWDEMO_UART_MSG_STATS = 6;
MMWDEMO_UART_MSG_DETECTED_POINTS_SIDE_INFO = 7;
MMWDEMO_UART_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP = 8;
MMWDEMO_UART_MSG_TEMPERATURE_STATS = 9;

%display('version 0.6');

% added by CI to remove ground clutter
global Z_GND_MAX
Z_GND_MAX = -0.5;

global Z_GND_MIN
Z_GND_MIN = -0.8; 


% below defines correspond to the mmw demo code
MAX_NUM_OBJECTS = 100;
OBJ_STRUCT_SIZE_BYTES = 16;
SIDE_INFO_STRUCT_SIZE_BYTES = 4;
global NUM_ANGLE_BINS
NUM_ANGLE_BINS = 64;

global STATS_SIZE_BYTES
STATS_SIZE_BYTES = 16;


global bytevec_log;
bytevec_log = [];

global readUartFcnCntr;
readUartFcnCntr = 0;

global ELEV_VIEW 
%ELEV_VIEW = 3;
ELEV_VIEW = 2; %xy demo
global EXIT_KEY_PRESSED
EXIT_KEY_PRESSED = 0;

global Params
global hndChirpParamTable
hndChirpParamTable = [];

global BYTE_VEC_ACC_MAX_SIZE
BYTE_VEC_ACC_MAX_SIZE = 2^15;
global bytevecAcc
bytevecAcc = zeros(BYTE_VEC_ACC_MAX_SIZE,1);
global bytevecAccLen
bytevecAccLen = 0;


global BYTES_AVAILABLE_FLAG
BYTES_AVAILABLE_FLAG = 0;

global BYTES_AVAILABLE_FCN_CNT
BYTES_AVAILABLE_FCN_CNT = 32*8;

adaptiveCalibrationFlag = 0;
    
%Setup the main figure
figHnd = figure(1);
clf(figHnd);
if platformType == hex2dec('a1642')
    set(figHnd,'Name','Texas Instruments - XWR16xx mmWave Demo Visualization','NumberTitle','off')
elseif platformType == hex2dec('a1443')
    set(figHnd,'Name','Texas Instruments - XWR14xx mmWave Demo Visualization','NumberTitle','off')
elseif platformType == hex2dec('a1842')
    set(figHnd,'Name','Texas Instruments - XWR18xx mmWave Demo Visualization','NumberTitle','off')
elseif platformType == hex2dec('a1843')
    set(figHnd,'Name','Texas Instruments - XWR1843 mmWave Demo Visualization','NumberTitle','off')
elseif platformType == hex2dec('a6843')
    set(figHnd,'Name','Texas Instruments - XWR6843 mmWave Demo Visualization','NumberTitle','off')
else
    set(figHnd,'Name','Texas Instruments - Unknown platform mmWave Demo Visualization','NumberTitle','off')
end
warning off MATLAB:ui:javaframe:PropertyToBeRemoved;
warning off MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame
jframe=get(figHnd,'javaframe');
jIcon=javax.swing.ImageIcon(strcat(pwd, '/ti_icon.gif'));
jframe.setFigureIcon(jIcon);
%set(figHnd, 'MenuBar', 'none');
set(figHnd, 'Color', [0.8 0.8 0.8]);
set(figHnd, 'KeyPressFcn', @myKeyPressFcn)
set(figHnd,'ResizeFcn',{@Resize_clbk, figHnd});


pause(0.00001);
set(jframe,'Maximized',1); 
pause(0.00001);


% btn = uicontrol('Style', 'checkbox', 'String', '3D view',...
%         'Units','normalized',...
%         'Position', [.8 .95 .15 .03],...
%         'Value', 1,...
%         'Callback', @checkbox1_Callback);   
uiCtrlPosition = [.01 2/3-0.04  .15 .03];
btn1 = uicontrol('Style', 'checkbox', 'String', 'Linear scale',...
        'Units','normalized',...
        'Position', uiCtrlPosition,...
        'Value', 1,...
        'Callback', @checkbox_LinRangeProf_Callback);       


uiCtrlPosition(2) = uiCtrlPosition(2) - 0.04;
btn3 = uicontrol('Style', 'checkbox', 'String', 'Record',...
        'Units','normalized',...
        'Position', uiCtrlPosition,...
        'Value', 0,...
        'Callback', @checkbox_Logging_Callback);       

uiCtrlPosition(2) = uiCtrlPosition(2) - 0.04;
btn5 = uicontrol('Style', 'checkbox', 'String', 'Azimuth Heatmap Nearfield Corrected',...
        'Units','normalized',...
        'Position', uiCtrlPosition,...
        'Value', 0,...
        'Callback', @checkbox_AzimuthHeatMapCorr_Callback);       

warning off MATLAB:griddata:DuplicateDataPoints

    
%Read Configuration file
cliCfgFileId = fopen(cliCfgFileName, 'r');
if cliCfgFileId == -1
    fprintf('File %s not found!\n', cliCfgFileName);
    return
else
    fprintf('Opening configuration file %s ...\n', cliCfgFileName);
end
cliCfg=[];
tline = fgetl(cliCfgFileId);
k=1;
while ischar(tline)
    cliCfg{k} = tline;
    tline = fgetl(cliCfgFileId);
    k = k + 1;
end
fclose(cliCfgFileId);

global StatsInfo
StatsInfo.interFrameProcessingTime = 0;
StatsInfo.transmitOutputTime = 0;
StatsInfo.interFrameProcessingMargin = 0;
StatsInfo.interChirpProcessingMargin = 0;
StatsInfo.interFrameCPULoad = 0;
StatsInfo.activeFrameCPULoad = 0;


global activeFrameCPULoad
global interFrameCPULoad
activeFrameCPULoad = zeros(100,1);
interFrameCPULoad = zeros(100,1);
global guiCPULoad
guiCPULoad = zeros(100,1);
global xwrSocLoad
xwrSocLoad = zeros(100,1);


global guiProcTime
guiProcTime = 0;

displayUpdateCntr =0;



%Parse CLI parameters
Params = parseCfg(cliCfg);
if (Params.advFrameCfg.numOfSubFrames > 4)
    fprintf('Invalid number of subframes\n');
    return;
end

Params.nearEndCorr = [];
if (Params.isAdvanceSubFrm == 1)
    numTotalLoops = Params.advFrameCfg.numOfSubFrames;
else
    numTotalLoops = 1;
end
for k = 1:numTotalLoops
    Params.nearEndCorr{k} = nearEndCorrectionCalc(Params, k);
end

log2ToLog10 = 20*log10(2);
if Params.dataPath.numTxAnt == 3
    log2Qformat = (4/3)*1/256;
else
    log2Qformat = 1/256;
end

global maxRngProfYaxis
global maxRangeProfileYaxisLin
global maxRangeProfileYaxisLog
maxRangeProfileYaxisLin = maxRangeProfileYaxis;
maxRangeProfileYaxisLog = log2(maxRangeProfileYaxis) * log2ToLog10;
global rangeProfileLinearScale
rangeProfileLinearScale = 1;
if rangeProfileLinearScale == 1
    maxRngProfYaxis = maxRangeProfileYaxisLin;
else
    maxRngProfYaxis = maxRangeProfileYaxisLog;
end


%Configure monitoring UART port 
sphandle = configureSport(comportSnum);

%Send Configuration Parameters to XWR16xx
%Open CLI port
if ischar(comportCliNum)
    comportCliNum=str2num(comportCliNum);
end
if loadCfg == 1
    spCliHandle = configureCliPort(comportCliNum);

    warning off; %MATLAB:serial:fread:unsuccessfulRead
    timeOut = get(spCliHandle,'Timeout');
    set(spCliHandle,'Timeout',1);
    tStart = tic;
    while 1
        fprintf(spCliHandle, ''); cc=fread(spCliHandle,100);
        cc = strrep(strrep(cc,char(10),''),char(13),'');
        if ~isempty(cc)
                break;
        end
        pause(0.1);
        toc(tStart);
    end
    set(spCliHandle,'Timeout', timeOut);
    warning on;
    
    
    %Send CLI configuration to XWR1xxx
    fprintf('Sending configuration to XWR1xxx %s ...\n', cliCfgFileName);
    for k=1:length(cliCfg)
        if isempty(strrep(strrep(cliCfg{k},char(9),''),char(32),''))
            continue;
        end
        if strcmp(cliCfg{k}(1),'%')
            continue;
        end
        fprintf(spCliHandle, cliCfg{k});
        fprintf('%s\n', cliCfg{k});
        for kk = 1:3
            cc = fgetl(spCliHandle);
            if strcmp(cc,'Done')
                fprintf('%s\n',cc);
                break;
            elseif ~isempty(strfind(cc, 'not recognized as a CLI command'))
                fprintf('%s\n',cc);
                return;
            elseif ~isempty(strfind(cc, 'Error'))
                fprintf('%s\n',cc);
                return;
            end
        end
%             pause(.5);
%         C = strsplit(cliCfg{k});
%         if strcmp(C{1},'sensorStop')
%             pause(5);
%         elseif strcmp(C{1},'flushCfg')
%             pause(5);
%         else
%             pause(.5);
%         end
        pause(0.2)
    end
    fclose(spCliHandle);
    delete(spCliHandle);
end
    
    
displayChirpParams(Params, figHnd);
byteVecIdx = 0
rangePlotYaxisMin = 1e10;
if (numTotalLoops > 1)
    % find max range
    rangeValues = zeros(1, numTotalLoops);
    for k = 1:numTotalLoops
        rangeValues(k) = Params.dataPath.rangeIdxToMeters(k) *...
            Params.dataPath.numRangeBins(k);
    end
    rangePlotXaxisMax = min(rangeValues);
else
    rangePlotXaxisMax = Params.dataPath.rangeIdxToMeters(1) *...
        Params.dataPath.numRangeBins(1);
end

% rangePlotXaxisMax must be defined as follows with values from the short range subframe
% rangePlotXaxisMax = Params.dataPath.rangeIdxToMeters * Params.dataPath.numRangeBins;
% previous code trying to find the largest value is not correct. Actually we should use
% the smallest range since we only want to display the Range Profile for the short distance
% subframe

tStart = tic;
tIdleStart = tic;
timeout_ctr = 0;
bytevec_cp_max_len = 2^17;
bytevec_cp = zeros(bytevec_cp_max_len,1);
bytevec_cp_len = 0;

packetNumberPrev = 0;
numOfSubFrames = 1;
numOfSubFramesPrev = 1;

global loggingEnable
loggingEnable = 0;
global fidLog;
fidLog = 0;


% start change
% modify the following idx<Plot> to the subframe number plus 1 (to account
% for MATLAB indexing) you desire for whichever plot in the GUI
% ex: to display subframe 0 for range profile, set idxRangeProfile to 1
global idxRangeProfile
idxRangeProfile = 2; % 8m 3D
global idxRangeDoppler
idxRangeDoppler = 1; % 50m 2D
global idxAzimHeatmap
idxAzimHeatmap = 1;
% end change

%Initalize figures
figIdx = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLOUD POINTS TAB GROUP %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (Params.enGUI.detectedObjects == 1)
    Params.figCloudPoint.position = [3/13 1/2 5/13 1/2];
    Params.figCloudPoint.hTabGroup = uitabgroup(figHnd, 'Position', Params.figCloudPoint.position);
    Params.figCloudPoint.htab1 = uitab(Params.figCloudPoint.hTabGroup, 'Title', 'Cloud Points');
    Params.figCloudPoint.hax1 = axes('Parent', Params.figCloudPoint.htab1);
    
    %Params.figCloudPoint.hax1 = subplot(Params.guiMonitor.numFigRow, Params.guiMonitor.numFigCol, figIdx);
    figIdx = figIdx + 1;
    
    color1 = ['b.' 'm.' 'g.' 'y.'];
    color2 = ['g.' 'y.' 'b.' 'm.'];
    % determine how many plots to have
    if (Params.isAdvanceSubFrm == 1)
        numPlots = Params.advFrameCfg.numOfSubFrames;
    else
        numPlots = 1;
    end
    
    hold off
    if Params.xyz3DPlot == 1
        hold on
        toLabel = [];
        labelName = {};
        for k = 1:numPlots
            Params.figCloudPoint.hplot(k) =...
                plot3(0, 0, 0, color1(2*k-1:2*k), 'MarkerSize', 10);
            toLabel = [toLabel Params.figCloudPoint.hplot(k)];
            labelName{k} = strcat('Subframe ', num2str(k-1));
        end
%         Params.figCloudPoint.hplot = plot3(0,0,0,'b.','MarkerSize',10);
        Params.detectedObjectsPlotHnd_y = plot3([0 0],[0 range_depth],[0 0],'-r');
        Params.detectedObjectsPlotHnd_x = plot3([-range_width range_width],[0 0],[0 0],'-r');
        Params.detectedObjectsPlotHnd_z = plot3([0 0],[0 0],[-range_width range_width],'-r');
        Params.detectedObjectsPlotHnd_000 = plot3(0,0,0,'ro','MarkerSize',5);
        
        if 0
            %Draw FoV for these angles:
            azimFovMin=-20;
            azimFovMax=-10;
            elevFovMax=15;
            elevFovMin=5;
            %plot3([0 range_depth*tand(azimFovMax)],    [0 range_depth], [0 0 ],'b');
            %plot3([0 range_depth*tand(azimFovMin)],   [0 range_depth], [0 0 ],'b');
            plot3([0 range_depth*tand(azimFovMax)],    [0 range_depth], [0 range_depth/cosd(azimFovMax)*tand(elevFovMax)],'b');
            plot3([0 range_depth*tand(azimFovMax)],    [0 range_depth], [0 range_depth/cosd(azimFovMax)*tand(elevFovMin)],'b');
            plot3([0 range_depth*tand(azimFovMin)],    [0 range_depth], [0 range_depth/cosd(azimFovMax)*tand(elevFovMax)],'b');
            plot3([0 range_depth*tand(azimFovMin)],    [0 range_depth], [0 range_depth/cosd(azimFovMax)*tand(elevFovMin)],'b');

            plot3([range_depth*tand(azimFovMax) range_depth*tand(azimFovMax)],    [ range_depth range_depth], [ range_depth/cosd(azimFovMax)*tand(elevFovMax) range_depth/cosd(azimFovMin)*tand(elevFovMin)],'b');
            plot3([range_depth*tand(azimFovMax) range_depth*tand(azimFovMin)],    [ range_depth range_depth], [ range_depth/cosd(azimFovMin)*tand(elevFovMin) range_depth/cosd(azimFovMin)*tand(elevFovMin)],'b');
            plot3([range_depth*tand(azimFovMin) range_depth*tand(azimFovMin)],    [ range_depth range_depth], [ range_depth/cosd(azimFovMax)*tand(elevFovMax) range_depth/cosd(azimFovMin)*tand(elevFovMin)],'b');
            plot3([range_depth*tand(azimFovMin) range_depth*tand(azimFovMax)],    [ range_depth range_depth], [ range_depth/cosd(azimFovMax)*tand(elevFovMax) range_depth/cosd(azimFovMax)*tand(elevFovMax)],'b');
             
            %plot3([0 0],              [0 range_depth],          [0 range_depth*tand(elevFovMax)],'g');
            %plot3([0 0],              [0 range_depth],          [0 range_depth*tand(elevFovMin) ],'g');
        end
        grid on          
        set(gca,'XLim',[-range_width range_width]);
        set(gca,'YLim',[0 range_depth]);      
        set(gca,'ZLim',[-range_width range_width]);
        xlabel('meters');                  
        ylabel('meters');
        title('3D Scatter Plot')
        view(ELEV_VIEW);
        rotate3d on
        if (Params.isAdvanceSubFrm == 1)
            legend(toLabel, labelName);
            legend('Location', 'bestoutside');
%             legend('Orientation', 'horizontal');
            legend('boxoff');
        end
    else
        hold on
        t = linspace(pi/6,5*pi/6,128);
        patch(Params.figCloudPoint.hax1, [0 range_depth*cos(t) 0], [0 range_depth*sin(t) 0],  [0.0 0.0 0.5] );
        axis equal                    

        axis(Params.figCloudPoint.hax1, [-range_width range_width 0 range_depth])
        xlabel(Params.figCloudPoint.hax1, 'Distance along lateral axis (meters)');                  
        ylabel(Params.figCloudPoint.hax1, 'Distance along longitudinal axis (meters)');
        plotGrid(Params.figCloudPoint.hax1, range_depth, range_grid_arc);
        title(Params.figCloudPoint.hax1, 'X-Y Scatter Plot')
        set(Params.figCloudPoint.hax1,'Color',[0 0 0.5]);
        toLabel = [];
        labelName = {};
        for k = 1:numPlots
            Params.figCloudPoint.hplot(k) =...
                plot(Params.figCloudPoint.hax1, 0, 0, color2(2*k-1:2*k),...
                'MarkerSize', 10);
            toLabel = [toLabel Params.figCloudPoint.hplot(k)];
            labelName{k} = strcat('Subframe ', num2str(k-1));
        end
        if (Params.isAdvanceSubFrm == 1)
            legend(toLabel, labelName);
            legend('Location', 'bestoutside');
%             legend('Orientation', 'horizontal');
            legend('boxoff');
        end
%         Params.figCloudPoint.hplot = plot(Params.figCloudPoint.hax1, 0,0,'g.', 'Marker', '.','MarkerSize',10);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RANGE PROFILE TAB GROUP %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (Params.enGUI.logMagRange == 1)
    
    % REVISIT
    Params.figRangeProfile.position = [8/13 1/2 5/13 1/2];
    Params.figRangeProfile.hTabGroup = uitabgroup(figHnd, 'Position', Params.figRangeProfile.position);
    Params.figRangeProfile.htab1 = uitab(Params.figRangeProfile.hTabGroup, 'Title', 'Range Profile');
    Params.figRangeProfile.hax1 = axes('Parent', Params.figRangeProfile.htab1);


    Params.figRangeProfile.dcm_obj = datacursormode(figHnd);
    set(Params.figRangeProfile.dcm_obj, 'UpdateFcn', @rp_dcm_callback);
    
    
    %Params.figRangeProfile.hax1 = subplot(Params.guiMonitor.numFigRow, Params.guiMonitor.numFigCol, figIdx);
    Params.figRangeProfile.update = 0;
    Params.figRangeProfile.hplot = plot(Params.figRangeProfile.hax1,...
        Params.dataPath.rangeIdxToMeters(idxRangeProfile) * [0:Params.dataPath.numRangeBins(idxRangeProfile)-1],...
        zeros(length(Params.dataPath.rangeIdxToMeters(idxRangeProfile) * [0:Params.dataPath.numRangeBins(idxRangeProfile)-1]), 3), '-');    
    hline = findobj(Params.figRangeProfile.hplot, 'type', 'line');
    set(hline(1),'LineStyle','-', 'color',[0 0 1])
    set(hline(2),'LineStyle','none', 'color',[1 0 0], 'Marker', 'x')
    set(hline(3),'LineStyle','-', 'color',[0 .5 0])
    figIdx = figIdx + 1;
    hold on;
    a=zeros(4,1);
    a(4) = maxRngProfYaxis;
    a(2) = rangePlotXaxisMax; 
    axis(a);
    xlabel('Range (m)');
    title('Range Profile');
    grid on;
    
    np = NaN*zeros(Params.dataPath.numRangeBins(idxRangeProfile), 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% RANGE_DOPPLER TAB GROUP %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (Params.enGUI.detectedObjects == 1) 
    
    % REVISIT
    Params.figRangeDoppler.hTabGroup = uitabgroup(figHnd, 'Position', [3/13 0 5/13 1/2]);
    %%%%%%%%%%%%%%%%%% TAB1: RANGE-DOPPLER CLOUD POINTS %%%%%%%%%%%%%%%%%%%
    Params.figRangeDoppler.htab(1) = uitab(Params.figRangeDoppler.hTabGroup, 'Title', 'Range-Doppler points');
    Params.figRangeDoppler.hax(1) = axes('Parent', Params.figRangeDoppler.htab(1));
    
    %Params.figRangeDoppler.hax(1) = subplot(Params.guiMonitor.numFigRow, Params.guiMonitor.numFigCol, figIdx);
    figIdx = figIdx + 1;
    hold off
    plot(0,0,'Color',[0 0 0.5])
    hold on
    set(gca,'Color',[0 0 0.5]);
    dopplerRange = Params.dataPath.dopplerResolutionMps(idxRangeDoppler) * Params.dataPath.numDopplerBins(idxRangeDoppler) / 2;
    if Params.extendedMaxVelocity.enable
        dopplerRange = dopplerRange * Params.dataPath.numTxAnt;
    end
    axis([0 range_depth -dopplerRange dopplerRange])
    xlabel('Range (meters)');
    ylabel('Doppler (m/s)');
    title('Doppler-Range Plot')
    grid on;
    set(gca,'Xcolor',[0.5 0.5 0.5]);
    set(gca,'Ycolor',[0.5 0.5 0.5]);
    Params.figRangeDoppler.hplot(1) = plot(0,0,'g.', 'Marker', '.','MarkerSize',5);
end 

if (Params.enGUI.rangeDopplerHeatMap == 1)
    
    % REVISIT
    %%%%%%%%%%%%%%%%%% TAB2: RANGE-DOPPLER HEATMAP %%%%%%%%%%%%%%%%%%%%%%
    Params.figRangeDoppler.htab(2) = uitab(Params.figRangeDoppler.hTabGroup, 'Title', 'Range-Doppler Heatmap');
    Params.figRangeDoppler.hax(2) = axes('Parent', Params.figRangeDoppler.htab(2));
    
    %Params.figRangeDoppler.hax(2) = subplot(Params.guiMonitor.numFigRow, Params.guiMonitor.numFigCol, figIdx);
    %figIdx = figIdx + 1;

    Params.figRangeDoppler.hplot(2) = ...
            surf(Params.dataPath.rangeIdxToMeters(idxRangeDoppler)*[0:Params.dataPath.numRangeBins(idxRangeDoppler)-1],...
            Params.dataPath.dopplerResolutionMps(idxRangeDoppler)*[-Params.dataPath.numDopplerBins(idxRangeDoppler)/2 : Params.dataPath.numDopplerBins(idxRangeDoppler)/2-1],...
            zeros(Params.dataPath.numDopplerBins(idxRangeDoppler), Params.dataPath.numRangeBins(idxRangeDoppler)));
    shading interp
    view(2);

    dopplerRange = Params.dataPath.dopplerResolutionMps(idxRangeDoppler) * Params.dataPath.numDopplerBins(idxRangeDoppler) / 2;
    axis(Params.figRangeDoppler.hax(2), ...
        [0 range_depth -Params.dataPath.dopplerResolutionMps(idxRangeDoppler)*Params.dataPath.numDopplerBins(idxRangeDoppler)/2 Params.dataPath.dopplerResolutionMps(idxRangeDoppler)*(Params.dataPath.numDopplerBins(idxRangeDoppler)/2-1)]);
    xlabel(Params.figRangeDoppler.hax(2), 'Range (meters)');
    ylabel(Params.figRangeDoppler.hax(2), 'Doppler (m/s)');              
    title(Params.figRangeDoppler.hax(2), 'Doppler-Range Heatmap')
        
    %%%%%%%%%%%%%%%%%% TAB3: RANGE-DOPPLER PRFILES %%%%%%%%%%%%%%%%%%%%%%
    Params.figRangeDoppler.htab(3) = uitab(Params.figRangeDoppler.hTabGroup, 'Title', 'Range-Doppler Profiles');
    Params.figRangeDoppler.hax(3) = axes('Parent', Params.figRangeDoppler.htab(3));
    
    Params.figRangeDoppler.hPlot3 = plot(Params.figRangeDoppler.hax(3), (0:Params.dataPath.numRangeBins(idxRangeDoppler)-1)*Params.dataPath.rangeIdxToMeters(idxRangeDoppler), zeros(Params.dataPath.numRangeBins(idxRangeDoppler),Params.dataPath.numDopplerBins(idxRangeDoppler))); 


    xlabel(Params.figRangeDoppler.hax(3),'Range (meters)');
    ylabel(Params.figRangeDoppler.hax(3),'Magnitude squared');              
    title(Params.figRangeDoppler.hax(3),'Range-Doppler Profiles')
    axis(Params.figRangeDoppler.hax(3), [0  Params.dataPath.numRangeBins(idxRangeDoppler)*Params.dataPath.rangeIdxToMeters(idxRangeDoppler) 0 maxRngProfYaxis]);
    set(Params.figRangeDoppler.hax(3), 'YLimMode', 'manual');
    Params.figRangeDoppler.update = 0;        
    grid on;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% STATIC AZIMUTH HEATMAP TAB GROUP %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (Params.enGUI.rangeAzimuthHeatMap == 1)
    
    % REVISIT
    %Range complex bins at zero Doppler all virtual (azimuth) antennas
    Params.figAzimuthHeatmap.position = [8/13 0 5/13 1/2];
    Params.figAzimuthHeatmap.hTabGroup = uitabgroup(figHnd, 'Position', Params.figAzimuthHeatmap.position);
    
    %%%%%%%%%%%%%%%%%% TAB1: STATIC AZIMUTH HEATMAP% %%%%%%%%%%%%%%%%%%%%%%
    Params.figAzimuthHeatmap.htab(1) = uitab(Params.figAzimuthHeatmap.hTabGroup, 'Title', 'Azimuth Heatmap');
    Params.figAzimuthHeatmap.hax(1) = axes('Parent', Params.figAzimuthHeatmap.htab(1));
    
    %Params.figAzimuthHeatmap.hax(1) = subplot(Params.guiMonitor.numFigRow, Params.guiMonitor.numFigCol, figIdx);
    figIdx = figIdx + 1;
    hold on;
    theta = asind([-NUM_ANGLE_BINS/2+1 : NUM_ANGLE_BINS/2-1]'*(2/NUM_ANGLE_BINS));
    range = [0:Params.dataPath.numRangeBins(idxAzimHeatmap)-1] * Params.dataPath.rangeIdxToMeters(idxAzimHeatmap);
    posX = range' * sind(theta');
    posY = range' * cosd(theta');
    xlin=linspace(-range_width,range_width,200);
    ylin=linspace(0,range_depth,200);
    view([0,90])
    set(Params.figAzimuthHeatmap.hax(1),'Color',[0 0 0.5]);
    axis(Params.figAzimuthHeatmap.hax(1),'equal')
    axis(Params.figAzimuthHeatmap.hax(1),[-range_width range_width 0 range_depth])
    xlabel(Params.figAzimuthHeatmap.hax(1), 'Distance along lateral axis (meters)');                  
    ylabel(Params.figAzimuthHeatmap.hax(1), 'Distance along longitudinal axis (meters)');
    title(Params.figAzimuthHeatmap.hax(1), 'Azimuth-Range Heatmap')

    
    
    [X,Y]=meshgrid(xlin,ylin);
    warning off
    QQ = zeros(Params.dataPath.numRangeBins(idxAzimHeatmap), Params.dataPath.numAzimuthBins -1);
    Z=griddata(posX,posY,QQ,X,Y,'linear');
    warning on
    Params.figAzimuthHeatmap.hplot(1) = surf(Params.figAzimuthHeatmap.hax(1), xlin,ylin,Z);
    shading INTERP
    view(Params.figAzimuthHeatmap.hax(1), [0,90])
    set(Params.figAzimuthHeatmap.hax(1), 'Color',[0 0 0.5]);
    axis(Params.figAzimuthHeatmap.hax(1), 'equal')
    axis(Params.figAzimuthHeatmap.hax(1), [-range_width range_width 0 range_depth])
    xlabel(Params.figAzimuthHeatmap.hax(1), 'Distance along lateral axis (meters)');                  
    ylabel(Params.figAzimuthHeatmap.hax(1), 'Distance along longitudinal axis (meters)');
    title(Params.figAzimuthHeatmap.hax(1), 'Azimuth-Range Heatmap')    
    
    Params.figAzimuthHeatmap.htab(2) = uitab(Params.figAzimuthHeatmap.hTabGroup, 'Title', 'Rx Antenna Angles');
    Params.figAzimuthHeatmap.hax(2) = axes('Parent', Params.figAzimuthHeatmap.htab(2));

    Params.figAzimuthHeatmap.htab(3) = uitab(Params.figAzimuthHeatmap.hTabGroup, 'Title', 'Rx Antenna Amplitudes');
    Params.figAzimuthHeatmap.hax(3) = axes('Parent', Params.figAzimuthHeatmap.htab(3));

    Params.figAzimuthHeatmap.htab(4) = uitab(Params.figAzimuthHeatmap.hTabGroup, 'Title', '2D Angle FFT Heatmap');
    Params.figAzimuthHeatmap.hax(4) = axes('Parent', Params.figAzimuthHeatmap.htab(4));
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% STATS CPU LOAD TAB GROUP%%%%%%%% %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (Params.enGUI.stats == 1)
    %%%%%%%%%%%%%%%%%% TAB1: STATIC AZIMUTH HEATMAP% %%%%%%%%%%%%%%%%%%%%%%
    Params.figStatsCPU.position = [0 0 3/13 1/3];
    Params.figStatsCPU.hTabGroup = uitabgroup(figHnd, 'Position', Params.figStatsCPU.position);
    Params.figStatsCPU.htab(1) = uitab(Params.figStatsCPU.hTabGroup, 'Title', 'CPU Load');
    Params.figStatsCPU.hax(1) = axes('Parent', Params.figStatsCPU.htab(1));
    
     %Params.figStatsCPU.hax(1) = subplot(Params.guiMonitor.numFigRow, Params.guiMonitor.numFigCol, figIdx);
     Params.figStatsCPU.hplot1 = plot(zeros(100,4));
     figIdx = figIdx + 1;
     hold on;
     xlabel('frames');                  
     ylabel('% CPU Load');
     axis([0 100 0 100])
     title('Active and Interframe CPU Load')
     plot([0 0 0; 0 0 0])
     legend('Interframe', 'Active frame', 'GUI', 'SOC', 'Location', 'northwest')
end

magicNotOkCntr=0;

if adaptiveCalibrationFlag
    cComp = ones(8,1);
    cCompMu = 1e-9;
    myFid = figure;
    myHax = axes('Parent', myFid);
    myFid1 = figure;
    myHax1 = axes('Parent', myFid1);
    myFid2 = figure;
    myHax2 = axes('Parent', myFid2);
    myHplot2 = plot(myHax2, zeros(8,1),zeros(8,1),'g.-');
    hold(myHax2,'on')
    plot([0 0],[-2 2],'k')
    plot([-2 2],[0 0],'k')
    viscircles([0 0],1,'Color','k','linewidth',1,'linestyle', ':')
    title(myHax2,'Compensation Coefficients');
    axis(myHax2,2*[-1 1 -1 1]);
    axis(myHax2,'square')
end

global sfIdx
sfIdx = 0;
%-------------------- Main Loop ------------------------
while (~EXIT_KEY_PRESSED)
    %Read bytes
    readUartCallbackFcn(sphandle, 0);
    
    if BYTES_AVAILABLE_FLAG == 1
        BYTES_AVAILABLE_FLAG = 0;
        %fprintf('bytevec_cp_len, bytevecAccLen = %d %d \n',bytevec_cp_len, bytevecAccLen)
        if (bytevec_cp_len + bytevecAccLen) < bytevec_cp_max_len
            bytevec_cp(bytevec_cp_len+1:bytevec_cp_len + bytevecAccLen) = bytevecAcc(1:bytevecAccLen);
            bytevec_cp_len = bytevec_cp_len + bytevecAccLen;
            bytevecAccLen = 0;
        else
            fprintf('Error: Buffer overflow, bytevec_cp_len, bytevecAccLen = %d %d \n',bytevec_cp_len, bytevecAccLen)
        end
    end
    
    bytevecStr = char(bytevec_cp);
    magicOk = 0;
    startIdx = strfind(bytevecStr', char([2 1 4 3 6 5 8 7]));
    if ~isempty(startIdx)
        if startIdx(1) > 1
            bytevec_cp(1: bytevec_cp_len-(startIdx(1)-1)) = bytevec_cp(startIdx(1):bytevec_cp_len);
            bytevec_cp_len = bytevec_cp_len - (startIdx(1)-1);
        end
        if bytevec_cp_len < 0
            fprintf('Error: %d %d \n',bytevec_cp_len, bytevecAccLen)
            bytevec_cp_len = 0;
        end

        totalPacketLen = sum(bytevec_cp(8+4+[1:4]) .* [1 256 65536 16777216]');                
        if bytevec_cp_len >= totalPacketLen
            magicOk = 1;
        else
            magicOk = 0;
        end
    end

    periodicity = 0;
    byteVecIdx = 0;
    if(magicOk == 1)
        %fprintf('OK, bytevec_cp_len = %d\n',bytevec_cp_len);
        if debugFlag
            fprintf('Frame Interval = %.3f sec,  ', toc(tStart));
        end
        tStart = tic;

        detectedObjectsPresent       = 0;
        logMagRangePresent           = 0;
        noiseProfilePresent          = 0;
        rangeAzimuthHeatMapPresent   = 0;
        rangeDopplerHeatMapPresent   = 0;
        statsPresent                 = 0;
        
        [Header, byteVecIdx] = getHeader(bytevec_cp, byteVecIdx);
        sfIdx = Header.subFrameNumber + 1;
        detObj.numObj = 0;
        % extrapolate profile used
        if (Params.isAdvanceSubFrm == 1)
            chirpStartIdx = Params.subFrameCfg(sfIdx).chirpStartIdx;
            profileIdx = Params.chirpCfg(chirpStartIdx+1).profileIdx;
            periodicity = Params.subFrameCfg(sfIdx).subFramePeriodicity;
        else
            profileIdx = 0;
            periodicity = Params.frameCfg.framePeriodicity;
        end
        
        for tlvIdx = 1:Header.numTLVs
            [tlv, byteVecIdx] = getTlv(bytevec_cp, byteVecIdx);
            switch tlv.type
                case MMWDEMO_UART_MSG_DETECTED_POINTS
                    if tlv.length >= OBJ_STRUCT_SIZE_BYTES
                        [detObj, byteVecIdx] = getDetObj(bytevec_cp, ...
                                                        byteVecIdx, ...
                                                        tlv.length, ...
                                                        Header.numDetectedObj);
                        detectedObjectsPresent       = 1;
                    end
                case MMWDEMO_UART_MSG_DETECTED_POINTS_SIDE_INFO
                    if tlv.length >= SIDE_INFO_STRUCT_SIZE_BYTES
                        [sideInfo, byteVecIdx] = getSideInfo(bytevec_cp, ...
                                                        byteVecIdx, ...
                                                        tlv.length, ...
                                                        Header.numDetectedObj);
                    end
                case MMWDEMO_UART_MSG_RANGE_PROFILE
                    [rp, byteVecIdx] = getRangeProfile(bytevec_cp, ...
                                                byteVecIdx, ...
                                                tlv.length, 1);
                    if Params.profileCfg(profileIdx+1).freqSlopeConst < 0
                        rp = circshift(flipud(rp),1);
                    end
                    logMagRangePresent           = 1;
                                                    
                case MMWDEMO_UART_MSG_NOISE_PROFILE
                    [np, byteVecIdx] = getRangeProfile(bytevec_cp, ...
                                                byteVecIdx, ...
                                                tlv.length, 0);
                    if Params.profileCfg(profileIdx+1).freqSlopeConst < 0
                        np = circshift(flipud(np),1);
                    end
                    noiseProfilePresent          = 1;
                case MMWDEMO_UART_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP
                        [Q, q, byteVecIdx] = getAzimElevStaticHeatMap(Params,...
                                                bytevec_cp, ...
                                                byteVecIdx, ...
                                                Params.dataPath.numTxAnt, ...
                                                Params.dataPath.numRxAnt,...
                                                Params.dataPath.numRangeBins(sfIdx),...
                                                NUM_ANGLE_BINS,...
                                                tlv.length);
                        plotAngleFFT(Params, q, Params.selectedRngIdx);
                        rangeAzimuthHeatMapPresent   = 1;
                case MMWDEMO_UART_MSG_AZIMUT_STATIC_HEAT_MAP
                        [Q, Qcorr, q, byteVecIdx] = getAzimuthStaticHeatMap(bytevec_cp, ...
                                                byteVecIdx, ...
                                                Params.dataPath.numTxAzimAnt, ...
                                                Params.dataPath.numRxAnt,...
                                                Params.dataPath.numRangeBins(sfIdx),...
                                                NUM_ANGLE_BINS,...
                                                Params.nearEndCorr, ...
                                                tlv.length);
                                           
                    rangeAzimuthHeatMapPresent   = 1;
                case MMWDEMO_UART_MSG_RANGE_DOPPLER_HEAT_MAP
                    [rangeDoppler, byteVecIdx] = getRangeDopplerHeatMap(bytevec_cp, ...
                                                byteVecIdx, ...
                                                Params.dataPath.numDopplerBins(sfIdx), ...
                                                Params.dataPath.numRangeBins(sfIdx), ...
                                                tlv.length);
                    rangeDopplerHeatMapPresent   = 1;
                    
                case MMWDEMO_UART_MSG_STATS
                    [StatsInfo, byteVecIdx] = getStatsInfo(bytevec_cp, ...
                                                byteVecIdx);
                     %fprintf('StatsInfo: %d, %d, %d %d \n', StatsInfo.interFrameProcessingTime, StatsInfo.transmitOutputTime, StatsInfo.interFrameProcessingMargin, StatsInfo.interChirpProcessingMargin);
                     displayUpdateCntr = displayUpdateCntr + 1;
                     interFrameCPULoad = [interFrameCPULoad(2:end); StatsInfo.interFrameCPULoad];
                     activeFrameCPULoad = [activeFrameCPULoad(2:end); StatsInfo.activeFrameCPULoad];
                     guiCPULoad = [guiCPULoad(2:end); 100*guiProcTime/periodicity];
                     socLoad = 100*((StatsInfo.interFrameProcessingMargin - StatsInfo.transmitOutputTime))/(1000*periodicity);
                     socLoad = max(0, socLoad);
                     socLoad = min(100, socLoad);
                     xwrSocLoad = [xwrSocLoad(2:end); 100 - socLoad];
                     if displayUpdateCntr == 40
                        UpdateDisplayTable(Params);
                        displayUpdateCntr = 0; %REMOVE
                     end
                    statsPresent   = 1;

                case MMWDEMO_UART_MSG_TEMPERATURE_STATS
                    byteVecIdx = byteVecIdx + tlv.length;
                    
                otherwise
                    byteVecIdx = byteVecIdx + tlv.length;
            end
        end

        byteVecIdx = Header.totalPacketLen;
        numOfSubFrames = max(numOfSubFrames, Header.subFrameNumber+1);

        frameNumber = Header.frameNumber * numOfSubFrames + Header.subFrameNumber;

        if ((frameNumber - packetNumberPrev) ~= 1) && ...
            (packetNumberPrev ~= 0) && ...
            (numOfSubFrames == numOfSubFramesPrev)
               fprintf('Error: Packets lost: %d, current frame num = %d \n', ...
                       (Header.frameNumber - packetNumberPrev - 1), ...
                       Header.frameNumber);
        end
        packetNumberPrev = frameNumber;
        numOfSubFramesPrev = numOfSubFrames;

        %Log detected objects
        if (loggingEnable == 1)
            if (Header.numDetectedObj > 0)
                fprintf(fidLog, '%d %d\n', Header.frameNumber, detObj.numObj);
                fprintf(fidLog, '%.3f ', detObj.x);
                fprintf(fidLog, '\n');
                fprintf(fidLog, '%.3f ', detObj.y);
                fprintf(fidLog, '\n');
                if Params.xyz3DPlot == 1
                    fprintf(fidLog, '%.3f ', detObj.z);
                    fprintf(fidLog, '\n');
                end
                fprintf(fidLog, '%.3f ', detObj.doppler);
                fprintf(fidLog, '\n');
            end
        end            

        if (detectedObjectsPresent == 1)
            if detObj.numObj > 0
                % Display Detected objects
                if Params.xyz3DPlot == 1
                    %Plot detected objects in 3D
                    set(Params.figCloudPoint.hplot(sfIdx),...
                        'Xdata', detObj.x,...
                        'Ydata', detObj.y,...
                        'Zdata', detObj.z);
                    title(Params.figCloudPoint.hax1,...
                        sprintf('3D Scatter Plot, %d objects', Header.numDetectedObj))
                else
                    %Plot detected objects in 2D
                    set(Params.figCloudPoint.hplot(sfIdx),...
                        'Xdata', detObj.x,...
                        'Ydata', detObj.y);
                    title(Params.figCloudPoint.hax1,...
                        sprintf('X-Y Scatter Plot, %d objects', Header.numDetectedObj))
                end
            else
                if Params.xyz3DPlot == 1
                    %Empty Plot detected objects in 3D
                    set(Params.figCloudPoint.hplot(sfIdx),...
                        'Xdata', [],...
                        'Ydata', [],...
                        'Zdata', []);
                    title(Params.figCloudPoint.hax1,...
                        sprintf('3D Scatter Plot, %d objects', Header.numDetectedObj))                    
                else
                    %Empty plot detected objects in 2D
                    set(Params.figCloudPoint.hplot(sfIdx),...
                        'Xdata', [],...
                        'Ydata', []);
                    title(Params.figCloudPoint.hax1,...
                        sprintf('X-Y Scatter Plot, %d objects', Header.numDetectedObj))
                end
            end
        end

        if (sfIdx == idxRangeProfile)
            if (logMagRangePresent == 1)
                if rangeProfileLinearScale == 1
                    rp = Params.dspFftScaleCompAll_lin(sfIdx) * 2.^(rp*log2Qformat);
                    %rp = 2.^(rp*log2Qformat);
                else
                    rp = rp*log2ToLog10*log2Qformat + Params.dspFftScaleCompAll_log(sfIdx);
                end

                %Plot range profile
                %subplot(Params.figRangeProfile.hax1);

                rpDet=NaN*zeros(size(rp));
                if (detectedObjectsPresent == 1)
                    if detObj.numObj > 0
                        rpDet=NaN*zeros(size(rp));
                        rangeIdx = round(sqrt(detObj.x.^2+detObj.y.^2+detObj.z.^2)/Params.dataPath.rangeIdxToMeters(sfIdx));
                        zeroDopIdx = rangeIdx(detObj.doppler==0);
                        zeroDopIdx = zeroDopIdx(zeroDopIdx < length(rp));
                        zeroDopIdx = zeroDopIdx(zeroDopIdx >= 0);
                        if ~isempty(zeroDopIdx)
                            rpDet(zeroDopIdx+1) = rp(zeroDopIdx+1); %matlab indexes from 1 (not zero)
                        end
                    end
                end
                if noiseProfilePresent == 1
                    if exist('np', 'var')
                        if rangeProfileLinearScale == 1
                            np = Params.dspFftScaleCompAll_lin(sfIdx) * 2.^(np*log2Qformat);
                        else
                            np = np*log2ToLog10*log2Qformat + Params.dspFftScaleCompAll_log(sfIdx);
                        end
                    end
                end
                if length(rp) ~= length(rpDet) || length(rp) ~= length(np)
                    fprintf('Error: Lengths of rp, rpDet, np = %d,%d, %d \n', length(rp), length(rpDet), length(np));
                else
                    set(Params.figRangeProfile.hplot, {'Ydata'}, num2cell([rp rpDet np].',2));
                end

                if Params.figRangeProfile.update
                    Params.figRangeProfile.update = 0;
                    a = axis(Params.figRangeProfile.hax1);
                    a(4) = maxRngProfYaxis;
                    axis(Params.figRangeProfile.hax1, a);
                end
            end
 
        end
        
        if (sfIdx == idxRangeDoppler)
            %detectedObjectsPresent = 0;
            if (detectedObjectsPresent == 1)
                if detObj.numObj > 0
                    set(Params.figRangeDoppler.hplot(1), 'Xdata', sqrt(detObj.x.^2+detObj.y.^2+detObj.z.^2), 'Ydata', detObj.doppler);                
                else
                    set(Params.figRangeDoppler.hplot(1), 'Xdata', [], 'Ydata', []);                
                end
            end    
        end
        
        if (sfIdx == idxAzimHeatmap)
            if rangeAzimuthHeatMapPresent == 1
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if (0)            
                    hold off
                    [aa,peakId] = max(rp);
                    %fprintf('Peak index = %d\n', peakId);
                    for peakPos = peakId;
                        plot(((angle(q(1:4,peakPos))) - angle(q(1,peakPos)))/pi*180,'bo-')
                        hold on
                        plot(((angle(q(5:8,peakPos))) - angle(q(1,peakPos)))/pi*180,'g.-')
                        ylabel('Phase (degrees)')
                        xlabel('Antenna number')
                    end
                    grid
                end

                %%%%%%%%%%%% UPDATE Rx Antenna Angles %%%%%%%%%%%%%%%%%%%%%%%%
                [aa,peakId] = max(rp(1:end)); %Search for peak 
                for peakPos = peakId;
                      hold(Params.figAzimuthHeatmap.hax(2),'off')
                      plot(Params.figAzimuthHeatmap.hax(2), 180/pi*(unwrap(angle(q(:, peakPos).*q(1, peakPos)'))),'bo-')
                      axis(Params.figAzimuthHeatmap.hax(2), [1 8 -180 180])

                      plot(Params.figAzimuthHeatmap.hax(3), ((abs(q(:, peakPos)))),'bo-')
                      axis(Params.figAzimuthHeatmap.hax(3), [1 8 0 25000])
                      title(Params.figAzimuthHeatmap.hax(3), sprintf('peakPosition = %d', peakPos)); 
                      if 0
                          %rangle = -1*[180/pi*((angle(q(:, peakPos)*q(1, peakPos)')))]'
                          rangle = -1*[180/pi*((angle(q(:, peakPos))))]'
                          mag = abs(q(:, peakPos))';
                          mag = mag ./ abs(mag).^2 * min(mag);
                          rmag = round(20*log10(mag) * 10)
                      end
                      %calCof = mag.*exp(1j*rangle/180*pi);
                      %calCof = [real(calCof); imag(calCof)];
                      %calCof = calCof(:)'

    %                  plot(Params.figAzimuthHeatmap.hax(2), (angle(q(:, peakPos))),'bo-')
                      hold(Params.figAzimuthHeatmap.hax(2),'on')
                      %plot(unwrap(angle(q(:, peakPos+1))),'go-')
                      %plot(unwrap(angle(q(:, peakPos-1))),'co-')
    %                   qCorr = q(:, peakPos);
    %                   qCorr(5:8) = qCorr(5:8) .* Params.nearEndCorr(1,peakId);
    %                   qCorrAngle = unwrap(angle(qCorr));
                     % plot(Params.figAzimuthHeatmap.hax(2), [5:8],qCorrAngle(5:8)-qCorrAngle(1),'r*-')
                      %plot([5:8],(angle(q(5:8, peakPos) .* Params.nearEndCorr(1,peakId))),'r*-');

                     % qInv = ifft(Qcorr(:,peakPos));
                     % qInv = qInv(1:8);
                     % plot(Params.figAzimuthHeatmap.hax(2), unwrap(angle(qInv))-angle(qInv(1)),'m^-')
    %                   axis(Params.figAzimuthHeatmap.hax(2), [1 8 -15 15])
    %                   grid(Params.figAzimuthHeatmap.hax(2), 'on');
    %                   title(Params.figAzimuthHeatmap.hax(2), sprintf('Antenna symbols - Angle(rx(5)/rx(4)): %.2f deg, peakPos=%d', 180/pi*angle(q(5,peakPos)/q(4,peakPos)), peakPos)); 
    %                   ylabel(Params.figAzimuthHeatmap.hax(2), 'Symbol angle (degrees)');
    %                   xlabel(Params.figAzimuthHeatmap.hax(2), 'Antenna number');
                end

                if adaptiveCalibrationFlag
                    x = q(:, peakPos);
                    [cComp, y, s] = rx_phase_compensation(x, cComp, cCompMu);               
                    hold(myHax,'off')
                    plot(myHax, x,'b.-');
                    hold(myHax,'on')
                    plot(myHax, y,'ko-');
                    plot(myHax, s,'r.-');
                    axis(myHax,12000*[-1 1 -1 1]);
                    axis(myHax,'square')
                    grid(myHax, 'on');

                    hold(myHax1,'off')
                    plot(myHax1, angle(x),'b.-');
                    hold(myHax1,'on')
                    plot(myHax1, angle(y),'ko-');

                    set(myHplot2, 'Xdata', real(cComp), 'Ydata', imag(cComp));
                end
                %%%%%%%%%% UPDATE AZIMUTH HEATMAP %%%%%%%%%%%%%%%%%%%%%%%        
                if Params.displayAzimuthHeatMapCorrected 
                    QQ=fftshift(abs(Qcorr),1);
                else
                    QQ=fftshift(abs(Q),1);
                end
                QQ=QQ.';
                QQ=QQ(:,2:end);

                warning off
                Z=griddata(posX,posY,QQ,X,Y,'linear');
                warning on
                set(Params.figAzimuthHeatmap.hplot(1), 'Zdata', Z);

            end
        end

        %%%%%%%%%%%%% UPDATE RANGE-DOPPLER TAB GROUP %%%%%%%%%%%%%%%%%%%%%%%%%
        if (sfIdx == idxRangeDoppler)
            if rangeDopplerHeatMapPresent == 1
                %%%%%%%%%%% Update Range-Doppler Heatmap %%%%%%%%%%%%%%
                set(Params.figRangeDoppler.hplot(2), 'Zdata', rangeDoppler);                

                %%%%%%%%%%% Update Range-Doppler Profiles
                if rangeProfileLinearScale == 1
                    rangeDoppler = Params.dspFftScaleCompAll_lin(sfIdx) * 2.^(rangeDoppler*log2Qformat);
                else
                    rangeDoppler = rangeDoppler*log2ToLog10*log2Qformat + Params.dspFftScaleCompAll_log(sfIdx);
                end
                set(Params.figRangeDoppler.hPlot3, {'Ydata'}, num2cell(rangeDoppler,2));                 
                if Params.figRangeDoppler.update == 1
                    Params.figRangeDoppler.update = 0;
                    a = axis(Params.figRangeDoppler.hax(3));
                    a(4) = maxRngProfYaxis;
                    axis(Params.figRangeDoppler.hax(3), a);
                end
            end
        end
        
        %%%%%%%%%%%%% UPDATE CPU TAB GROUP %%%%%%%%%%%%%%%%%%%%%%%%%
        if (statsPresent == 1)
            set(Params.figStatsCPU.hplot1, {'Ydata'}, num2cell([interFrameCPULoad activeFrameCPULoad guiCPULoad, xwrSocLoad]',2));
        end
        
        guiProcTime = round(toc(tStart) * 1e3);
        if debugFlag
            fprintf('processing time %f secs \n',toc(tStart));
        end

    else
        magicNotOkCntr = magicNotOkCntr + 1;
        %fprintf('Magic word not found! cntr = %d\n', magicNotOkCntr);
    end

    %Remove processed data
    if byteVecIdx > 0
        shiftSize = byteVecIdx;
        bytevec_cp(1: bytevec_cp_len-shiftSize) = bytevec_cp(shiftSize+1:bytevec_cp_len);
        bytevec_cp_len = bytevec_cp_len - shiftSize;
        if bytevec_cp_len < 0
            fprintf('Error: bytevec_cp_len < bytevecAccLen, %d %d \n', bytevec_cp_len, bytevecAccLen)
            bytevec_cp_len = 0;
        end
    end
    if bytevec_cp_len > (bytevec_cp_max_len * 7/8)
        bytevec_cp_len = 0;
    end

    tIdleStart = tic;

    pause(0.001);
   
    
    if(toc(tIdleStart) > 2*periodicity/1000)
        timeout_ctr=timeout_ctr+1;
        if debugFlag == 1
            fprintf('Timeout counter = %d\n', timeout_ctr);
        end
        tIdleStart = tic;
    end
end
%close and delete handles before exiting
close(1); % close figure
fclose(sphandle); %close com port
delete(sphandle);
return

function [] = readUartCallbackFcn(obj, event)
global bytevecAcc;
global bytevecAccLen;
global readUartFcnCntr;
global BYTES_AVAILABLE_FLAG
global BYTES_AVAILABLE_FCN_CNT
global BYTE_VEC_ACC_MAX_SIZE

bytesToRead = get(obj,'BytesAvailable');
if(bytesToRead == 0)
    return;
end

[bytevec, byteCount] = fread(obj, bytesToRead, 'uint8');

if bytevecAccLen + length(bytevec) < BYTE_VEC_ACC_MAX_SIZE * 3/4
    bytevecAcc(bytevecAccLen+1:bytevecAccLen+byteCount) = bytevec;
    bytevecAccLen = bytevecAccLen + byteCount;
else
    bytevecAccLen = 0;
end

readUartFcnCntr = readUartFcnCntr + 1;
BYTES_AVAILABLE_FLAG = 1;
return

function [] = dispError()
disp('error!');
return

function [sphandle] = configureSport(comportSnum)
    global BYTES_AVAILABLE_FCN_CNT;

    if ~isempty(instrfind('Type','serial'))
        disp('Serial port(s) already open. Re-initializing...');
        delete(instrfind('Type','serial'));  % delete open serial ports.
    end
    comportnum_str=['COM' num2str(comportSnum)]
    sphandle = serial(comportnum_str,'BaudRate',921600);
    set(sphandle,'InputBufferSize', 2^16);
    set(sphandle,'Timeout',10);
    set(sphandle,'ErrorFcn',@dispError);
    set(sphandle,'BytesAvailableFcnMode','byte');
    set(sphandle,'BytesAvailableFcnCount', 2^16+1);%BYTES_AVAILABLE_FCN_CNT);
    set(sphandle,'BytesAvailableFcn',@readUartCallbackFcn);
    fopen(sphandle);
return

function [sphandle] = configureCliPort(comportPnum)
    %if ~isempty(instrfind('Type','serial'))
    %    disp('Serial port(s) already open. Re-initializing...');
    %    delete(instrfind('Type','serial'));  % delete open serial ports.
    %end
    comportnum_str=['COM' num2str(comportPnum)]
    sphandle = serial(comportnum_str,'BaudRate',115200);
    set(sphandle,'Parity','none')    
    set(sphandle,'Terminator','LF')    

    
    fopen(sphandle);
return

function []=ComputeCR_SNR(noiseEnergy,rp,Params)

global sfIdx

cr_range=floor([1.0 5.0]/Params.dataPath.rangeIdxToMeters(sfIdx));

[maxVal  max_idx]=max(rp(cr_range(1): cr_range(2)));

noiseSigma=sqrt(noiseEnergy/(256*2*8));

noiseSigma_dB=10*log10(abs(noiseSigma));

maxValdB=10*maxVal/(log2(10)*2^9);

SNRdB=maxValdB-noiseSigma_dB;
dist= (max_idx+cr_range(1)-1)*Params.dataPath.rangeIdxToMeters(sfIdx);
%[SNRdB dist]
%keyboard
return


% function checkbox1_Callback(hObject, eventdata, handles)
% global ELEV_VIEW
% % hObject    handle to checkbox1 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Hint: get(hObject,'Value') returns toggle state of checkbox1
% 
% if (get(hObject,'Value') == get(hObject,'Max'))
%   ELEV_VIEW = 3;
% else
%   ELEV_VIEW = 2;
% end

function checkbox_LinRangeProf_Callback(hObject, eventdata, handles)
global rangeProfileLinearScale
global Params
if (get(hObject,'Value') == get(hObject,'Max'))
  rangeProfileLinearScale = 1;
else
  rangeProfileLinearScale = 0;
end
global maxRngProfYaxis
global maxRangeProfileYaxisLin
global maxRangeProfileYaxisLog
if rangeProfileLinearScale == 1
    maxRngProfYaxis = maxRangeProfileYaxisLin;
else
    maxRngProfYaxis = maxRangeProfileYaxisLog;
end

Params.figRangeProfile.update = 1;
Params.figRangeDoppler.update = 1;
return




function checkbox_AzimuthHeatMapCorr_Callback(hObject, eventdata, handles)
global Params

if (get(hObject,'Value') == get(hObject,'Max'))
    Params.displayAzimuthHeatMapCorrected = 1;
else
    Params.displayAzimuthHeatMapCorrected = 0;
end
return

    


function checkbox_Logging_Callback(hObject, eventdata, handles)
global loggingEnable
global fidLog;

if (get(hObject,'Value') == get(hObject,'Max'))
    loggingEnable = 1;
    fid = fopen('lognum.dat', 'r');
    if fid ~= -1
        logNum = fscanf(fid, '%d');
        fclose(fid);
    else
        logNum = 0;
    end
    logNum = logNum +1;
    fid = fopen('lognum.dat', 'w');
    fprintf(fid, '%d\n', logNum);
    fclose(fid);
    fidLog = fopen(sprintf('log_%03d.dat',logNum),'w');
else
    loggingEnable = 0;
    if fidLog ~= 0
        fclose(fidLog);
    end

end
return
%Display Chirp parameters in table on screen
function displayChirpParams(Params, figHnd)
global hndChirpParamTable
global StatsInfo
global guiProcTime
    if (hndChirpParamTable ~= 0)
        delete(hndChirpParamTable);
        hndChirpParamTable = [];
    end
    
    if (Params.isAdvanceSubFrm == 1)
        numTotalLoops = Params.advFrameCfg.numOfSubFrames;
    else
        numTotalLoops = 1;
    end

    columnname =   {'___________Parameter (Units)___________', 'Value'};
    columnformat = {'char', 'numeric'};
    
    tabgp = uitabgroup(figHnd);
    table = zeros(1, numTotalLoops);
    
    for k = 1:numTotalLoops
        if (Params.isAdvanceSubFrm == 1)
            chirpStartIdx = Params.subFrameCfg(k).chirpStartIdx;
            profileIdx = Params.chirpCfg(chirpStartIdx+1).profileIdx;
            periodicity = Params.subFrameCfg(k).subFramePeriodicity;
            tabLabel = ['Subframe ' num2str(k-1)];
        else
            profileIdx = 0;
            periodicity = Params.frameCfg.framePeriodicity;
            tabLabel = 'Frame 0';
        end
        profileCfg = Params.profileCfg(profileIdx+1);
        
        dat =  {'Start Frequency (Ghz)', profileCfg.startFreq;...
                'Slope (MHz/us)', profileCfg.freqSlopeConst;...   
                'Samples per chirp', profileCfg.numAdcSamples;...
                'Chirps per frame',  Params.dataPath.numChirpsPerFrame(k);...
                'Sampling rate (Msps)', profileCfg.digOutSampleRate / 1000;...
                'Bandwidth (GHz)', abs(profileCfg.freqSlopeConst) * profileCfg.numAdcSamples /...
                                   profileCfg.digOutSampleRate;...
                'Range resolution (m)', Params.dataPath.rangeResolutionMeters(k);...
                'Range step (m)', Params.dataPath.rangeIdxToMeters(k);...
                'Velocity resolution (m/s)', Params.dataPath.dopplerResolutionMps(k);...
                'Number of Tx (MIMO)', Params.dataPath.numTxAnt;...
                'Frame periodicity (msec)', periodicity;...
                'InterFrameProcessingTime (usec)', StatsInfo.interFrameProcessingTime; ...
                'transmitOutputTime (usec)', StatsInfo.transmitOutputTime; ...
                'interFrameProcessingMargin (usec)', StatsInfo.interFrameProcessingMargin; ...
                'InterChirpProcessingMargin (usec)', StatsInfo.interChirpProcessingMargin; ...
                'GuiProcTime (msec)', guiProcTime;            
                };
        tab = uitab(tabgp, 'Title', tabLabel);
        table(k) = uitable(tab, 'Units', 'normalized',...
            'Data', dat,...
            'ColumnName', columnname,...
            'ColumnFormat', columnformat,...
            'ColumnWidth', 'auto',...
            'RowName', []);
    end

    tPos = get(table(1), 'Extent');
    if(tPos(4) > 1/3)
        tPos(4) = 1/3;
    end
    if(tPos(3) > (3/13 - 0.01*2))
        tPos(3) = (3/13 - 0.01*2);
    end
    tPos(1) = .01;%3/13 / 2 - tPos(3)/2;
    tPos(2) = 2/3 + 1/3 / 2 - tPos(4)/2;
    set(tabgp, 'Position',tPos);
    for k = 1:length(table)
        set(table(k), 'Position', [0 0 1 1]);
    end

    hndChirpParamTable = table;
return

function [row] = getDisplayTableStatRow(dat)
[nRow, nCol] = size(dat);
for row = 1:nRow
    if ~isempty(strfind(dat{row,1},'InterFrameProcessingTime'))
        break;
    end
end
return

function UpdateDisplayTable(Params)
global hndChirpParamTable
global StatsInfo
global guiProcTime
global Header
global sfIdx
    t = hndChirpParamTable(sfIdx);
    dat = get(t, 'Data');
    row = getDisplayTableStatRow(dat);
    dat{row, 2} = StatsInfo.interFrameProcessingTime;
    dat{row+1, 2} = StatsInfo.transmitOutputTime;
    dat{row+2, 2} = StatsInfo.interFrameProcessingMargin;
    dat{row+3, 2} = StatsInfo.interChirpProcessingMargin;
    dat{row+4, 2} = guiProcTime;
    set(t,'Data', dat);
return

%Read relevant CLI parameters and store into P structure
function [P] = parseCfg(cliCfg)
global TOTAL_PAYLOAD_SIZE_BYTES
global MAX_NUM_OBJECTS
global OBJ_STRUCT_SIZE_BYTES
global platformType
global STATS_SIZE_BYTES
global NUM_ANGLE_BINS
global gPlatform

    P=[];
    P.extendedMaxVelocity.enable = 0;
    P.extendedMaxVelocity.scheme = 0;
    P.isAdvanceSubFrm = 0;
    P.rxChannelMeasurementMode = 0;
    P.rangeBias = 0;
    P.displayAzimuthHeatMapCorrected = 0;
    P.nearFieldCfg.enable = 0;

    P.selectedRngIdx = -1;
    
   
    offset = 0;
    P.xyz3DPlot = 0;
    P.dataPath.numTxAzimAnt = 0;
    P.dataPath.txAntOrder = [];
    
    P.profileCfg = [];
    numProfileCfg = 0;
    P.chirpCfg = [];
    numChirpCfg = 0;
    P.subFrameCfg = [];
    numSubFrameCfg = 0;
    
    P.guiMonitor = [];
    sharedGUICfg = 0;
    
    P.enGUI = [];
    P.enGUI.detectedObjects = 0;
    P.enGUI.logMagRange = 0;
    P.enGUI.rangeAzimuthHeatMap = 0;
    P.enGUI.rangeDopplerHeatMap = 0;
    P.enGUI.stats = 0;
    
    for k=1:length(cliCfg)
        C = strsplit(cliCfg{k});
        if strcmp(C{1},'channelCfg')
            P.channelCfg.txChannelEn = str2num(C{3});
            if strcmpi(gPlatform, 'xwr16xx')
                P.dataPath.numTxAzimAnt = bitand(bitshift(P.channelCfg.txChannelEn,0),1) +...
                                          bitand(bitshift(P.channelCfg.txChannelEn,-1),1);
                P.dataPath.numTxElevAnt = 0;
                P.xyz3DPlot = 0;
                offset = 1;
            elseif strcmpi(gPlatform, 'xwr14xx')
                P.dataPath.numTxAzimAnt = bitand(bitshift(P.channelCfg.txChannelEn,0),1) +...
                                          bitand(bitshift(P.channelCfg.txChannelEn,-2),1);
                P.dataPath.numTxElevAnt = bitand(bitshift(P.channelCfg.txChannelEn,-1),1);
                if P.dataPath.numTxElevAnt > 0
                    P.xyz3DPlot = 1;
                end
                offset = 0;
            elseif strcmpi(gPlatform, 'xwr1843') ||...
                   strcmpi(gPlatform, 'xwr6843')
                P.dataPath.numTxAzimAnt = bitand(bitshift(P.channelCfg.txChannelEn,0),1) +...
                                          bitand(bitshift(P.channelCfg.txChannelEn,-2),1);
                P.dataPath.numTxElevAnt = bitand(bitshift(P.channelCfg.txChannelEn,-1),1);
                if P.dataPath.numTxElevAnt > 0
                    P.xyz3DPlot = 1;
                end
                offset = 1;
            elseif strcmpi(gPlatform, 'xwr18xx')
                P.dataPath.numTxAzimAnt = bitand(bitshift(P.channelCfg.txChannelEn,0),1) +...
                                          bitand(bitshift(P.channelCfg.txChannelEn,-1),1);
                P.dataPath.numTxElevAnt = bitand(bitshift(P.channelCfg.txChannelEn,-2),1);
                if P.dataPath.numTxElevAnt > 0
                    P.xyz3DPlot = 1;
                end
                offset = 0;
            elseif strcmpi(gPlatform, 'xwr6843aop')
%                 P.dataPath.numTxAzimAnt = bitand(bitshift(P.channelCfg.txChannelEn,0),1) +...
%                                           bitand(bitshift(P.channelCfg.txChannelEn,-1),1) +...
%                                           bitand(bitshift(P.channelCfg.txChannelEn,-2),1);
                P.dataPath.numTxElevAnt = 0;
                P.xyz3DPlot = 1; %always 3D plot
                offset = 1;
             else
                fprintf('Unknown platform \n');
                return
            end
            P.channelCfg.rxChannelEn = str2num(C{2});
            P.dataPath.numRxAnt = bitand(bitshift(P.channelCfg.rxChannelEn,0),1) +...
                                  bitand(bitshift(P.channelCfg.rxChannelEn,-1),1) +...
                                  bitand(bitshift(P.channelCfg.rxChannelEn,-2),1) +...
                                  bitand(bitshift(P.channelCfg.rxChannelEn,-3),1);
            P.dataPath.numTxAnt = P.dataPath.numTxElevAnt + P.dataPath.numTxAzimAnt;
                                
        elseif strcmp(C{1},'dataFmt')
        elseif strcmp(C{1},'profileCfg')
            profileNum = str2num(C{2});
            P.profileCfg(profileNum+1).startFreq = str2num(C{3});
            P.profileCfg(profileNum+1).idleTime =  str2num(C{4});
            P.profileCfg(profileNum+1).rampEndTime = str2num(C{6});
            P.profileCfg(profileNum+1).freqSlopeConst = str2num(C{9});
            P.profileCfg(profileNum+1).numAdcSamples = str2num(C{11});
            P.profileCfg(profileNum+1).digOutSampleRate = str2num(C{12}); %uints: ksps
            
            numProfileCfg = numProfileCfg + 1;
        elseif strcmp(C{1},'chirpCfg')
            if strcmpi(gPlatform, 'xwr6843aop')
                P.dataPath.txAntOrder(str2double(C{2})+1) = log2(str2double(C{9}));
                P.dataPath.numTxAzimAnt = P.dataPath.numTxAzimAnt + 1;
                P.dataPath.numTxAnt = P.dataPath.numTxAzimAnt;
            end
            chirpNum = str2num(C{2});
            P.chirpCfg(chirpNum+1).profileIdx = str2num(C{4});
            
            numChirpCfg = numChirpCfg + 1;
        elseif strcmp(C{1},'frameCfg')
            P.frameCfg.chirpStartIdx = str2num(C{2});
            P.frameCfg.chirpEndIdx = str2num(C{3});
            P.frameCfg.numLoops = str2num(C{4});
            P.frameCfg.numFrames = str2num(C{5});
            P.frameCfg.framePeriodicity = str2num(C{6});
        elseif strcmp(C{1}, 'advFrameCfg')
            P.advFrameCfg.numOfSubFrames = str2num(C{2});
            P.advFrameCfg.forceProfile = str2num(C{3});
            P.advFrameCfg.numFrames = str2num(C{4});
            P.advFrameCfg.triggerSelect = str2num(C{5});
            P.advFrameCfg.frameTrigDelay = str2num(C{6});
        elseif strcmp(C{1}, 'subFrameCfg')
            subFrameNum = str2num(C{2});
            P.subFrameCfg(subFrameNum+1).forceProfileIdx = str2num(C{3});
            P.subFrameCfg(subFrameNum+1).chirpStartIdx = str2num(C{4});
            P.subFrameCfg(subFrameNum+1).numOfChirps = str2num(C{5});
            P.subFrameCfg(subFrameNum+1).numLoops = str2num(C{6});
            P.subFrameCfg(subFrameNum+1).burstPeriodicity = str2num(C{7});
            P.subFrameCfg(subFrameNum+1).chirpStartIdxOffset = str2num(C{8});
            P.subFrameCfg(subFrameNum+1).numOfBurst = str2num(C{9});
            P.subFrameCfg(subFrameNum+1).numOfBurstLoops = str2num(C{10});
            P.subFrameCfg(subFrameNum+1).subFramePeriodicity = str2num(C{11});
            
            numSubFrameCfg = numSubFrameCfg + 1;
        elseif strcmp(C{1}, 'guiMonitor')
%             P.guiMonitor.detectedObjects = str2num(C{offset+2});
%             P.guiMonitor.logMagRange = str2num(C{offset+3});
%             P.guiMonitor.noiseProfile = str2num(C{offset+4});
%             P.guiMonitor.rangeAzimuthHeatMap = str2num(C{offset+5});
%             P.guiMonitor.rangeDopplerHeatMap = str2num(C{offset+6});
%             P.guiMonitor.stats = str2num(C{offset+7});
%             P.guiMonitor.detectedObjects = 1;
%             P.guiMonitor.logMagRange = 1;
%             P.guiMonitor.noiseProfile = 1;
%             P.guiMonitor.rangeAzimuthHeatMap = 1;
%             P.guiMonitor.rangeDopplerHeatMap = 1;
%             P.guiMonitor.stats = 1;
            subFrameNum = str2num(C{2});
            if (subFrameNum == -1)
                subFrameNum = 0;
                sharedGUICfg = 1;
            end
            P.guiMonitor(subFrameNum+1).subFrameIdx = str2num(C{2});
            P.guiMonitor(subFrameNum+1).detectedObjects = str2num(C{3});
            if (P.guiMonitor(subFrameNum+1).detectedObjects == 1)
                P.enGUI.detectedObjects = 1;
            end
            P.guiMonitor(subFrameNum+1).logMagRange = str2num(C{4});
            if (P.guiMonitor(subFrameNum+1).logMagRange == 1)
                P.enGUI.logMagRange = 1;
            end
            P.guiMonitor(subFrameNum+1).noiseProfile = str2num(C{5});
            P.guiMonitor(subFrameNum+1).rangeAzimuthHeatMap = str2num(C{6});
            if (P.guiMonitor(subFrameNum+1).rangeAzimuthHeatMap == 1)
                P.enGUI.rangeAzimuthHeatMap = 1;
            end
            P.guiMonitor(subFrameNum+1).rangeDopplerHeatMap = str2num(C{7});
            if (P.guiMonitor(subFrameNum+1).rangeDopplerHeatMap == 1)
                P.enGUI.rangeDopplerHeatMap = 1;
            end
            P.guiMonitor(subFrameNum+1).stats = str2num(C{8});
            if (P.guiMonitor(subFrameNum+1).stats == 1)
                P.enGUI.stats = 1;
            end
        elseif strcmp(C{1},'measureRangeBiasAndRxChanPhase')
            if str2num(C{2}) == 1
            	P.rxChannelMeasurementMode = 1;
            end
        elseif strcmp(C{1},'compRangeBiasAndRxChanPhase')
            P.rangeBias = str2num(C{2});
        elseif strcmp(C{1},'nearFieldCfg')
            P.nearFieldCfg.enable = str2num(C{offset+2});
        elseif strcmp(C{1},'extendedMaxVelocity')
            P.extendedMaxVelocity.enable = str2num(C{offset+2});
            P.extendedMaxVelocity.scheme = str2num(C{offset+2});
        elseif strcmp(C{1},'dfeDataOutputMode')
            P.dfeDataOutputMode = str2num(C{2});
            if P.dfeDataOutputMode == 3
                P.isAdvanceSubFrm = 1;
            elseif P.dfeDataOutputMode == 1
                P.isAdvanceSubFrm = 0;
            end
        end
    end
            
    
    if (P.isAdvanceSubFrm == 1)
        numTotalLoops = P.advFrameCfg.numOfSubFrames;
    else
        numTotalLoops = 1;
    end
    
    P.dataPath.numChirpsPerFrame = zeros(1, numTotalLoops);
    P.dataPath.numDopplerChirps = zeros(1, numTotalLoops);
    P.dataPath.numDopplerBins = zeros(1, numTotalLoops);
    P.dataPath.numRangeBins = zeros(1, numTotalLoops);
    P.dataPath.rangeResolutionMeters = zeros(1, numTotalLoops);
    P.dataPath.rangeIdxToMeters = zeros(1, numTotalLoops);
    P.dataPath.dopplerResolutionMps = zeros(1, numTotalLoops);
    P.dspFFTScaleCompAll_lin = zeros(1, numTotalLoops);
    P.dspFFTScaleCompAll_log = zeros(1, numTotalLoops);
    
    for k = 1:numTotalLoops
        if (P.isAdvanceSubFrm == 1)
            calcChirpsPerFrame =...
                P.subFrameCfg(k).numOfChirps *...
                P.subFrameCfg(k).numLoops;
            
            chirpStartIdx = P.subFrameCfg(k).chirpStartIdx;
            profileIdx = P.chirpCfg(chirpStartIdx+1).profileIdx;
        else
            calcChirpsPerFrame = (P.frameCfg.chirpEndIdx -...
                P.frameCfg.chirpStartIdx + 1) * P.frameCfg.numLoops;
            
            profileIdx = 0;
        end
        
        P.dataPath.numChirpsPerFrame(k) = calcChirpsPerFrame;
        P.dataPath.numDopplerChirps(k) =...
            P.dataPath.numChirpsPerFrame(k) / P.dataPath.numTxAnt;
        P.dataPath.numDopplerBins(k) =...
            2^ceil(log2(P.dataPath.numDopplerChirps(k)));
        
        % minimum number of Doppler bins is 8
        if (P.dataPath.numDopplerBins(k) <= 4)
            P.dataPath.numDopplerBins(k) = 8;
        end
        
        P.dataPath.numRangeBins(k) = pow2roundup(P.profileCfg(profileIdx+1).numAdcSamples);
        P.dataPath.rangeResolutionMeters(k) = 3e8 *...
            P.profileCfg(profileIdx+1).digOutSampleRate * 1e3 /...
            (2 * abs(P.profileCfg(profileIdx+1).freqSlopeConst) * 1e12 *...
            P.profileCfg(profileIdx+1).numAdcSamples);
        if (P.profileCfg(profileIdx+1).startFreq >= 76)
            CLI_FREQ_SCALE_FACTOR = (3.6); % 77 GHz
        else
            CLI_FREQ_SCALE_FACTOR = (2.7); % 60 GHz
        end
        mmwFreqSlopeConst = fix(P.profileCfg(profileIdx+1).freqSlopeConst *...
            (2^26) / ((CLI_FREQ_SCALE_FACTOR * 1e3) * 900.0));
        P.dataPath.rangeIdxToMeters(k) = 3e8 *...
            P.profileCfg(profileIdx+1).digOutSampleRate * 1e3 /...
            (2 * abs(mmwFreqSlopeConst) *  ((CLI_FREQ_SCALE_FACTOR * 1e3 * 900) /...
            (2^26))* 1e12 * P.dataPath.numRangeBins(k));
        startFreqConst = fix(P.profileCfg(profileIdx+1).startFreq *...
            (2^26) / CLI_FREQ_SCALE_FACTOR);
        P.dataPath.dopplerResolutionMps(k) = 3e8 /...
            (2 * startFreqConst / 67108864 * CLI_FREQ_SCALE_FACTOR * 1e9 *...
            (P.profileCfg(profileIdx+1).idleTime + P.profileCfg(profileIdx+1).rampEndTime) *...
            1e-6 * P.dataPath.numDopplerBins(k) * P.dataPath.numTxAnt);
        
        %Calculate monitoring packet size
        tlSize = 8 %TL size 8 bytes
        TOTAL_PAYLOAD_SIZE_BYTES = 32; % size of header
        
        if (sharedGUICfg == 1)
            iGUI = 1;
        else
            iGUI = k;
        end
        
        P.guiMonitor(iGUI).numFigures = 1;%One figure for numerical parameters
        if (P.guiMonitor(iGUI).detectedObjects == 1 &&...
                P.guiMonitor(iGUI).rangeDopplerHeatMap == 1)
            TOTAL_PAYLOAD_SIZE_BYTES = TOTAL_PAYLOAD_SIZE_BYTES +...
                OBJ_STRUCT_SIZE_BYTES * MAX_NUM_OBJECTS + tlSize;
            P.guiMonitor(iGUI).numFigures = P.guiMonitor.numFigures(iGUI) + 1; %1 plots: X/Y plot
        end 
        if (P.guiMonitor(iGUI).detectedObjects == 1 &&...
                P.guiMonitor(iGUI).rangeDopplerHeatMap ~= 1)
            TOTAL_PAYLOAD_SIZE_BYTES = TOTAL_PAYLOAD_SIZE_BYTES +...
                OBJ_STRUCT_SIZE_BYTES * MAX_NUM_OBJECTS + tlSize;
            P.guiMonitor(iGUI).numFigures = P.guiMonitor(iGUI).numFigures + 2; %2 plots: X/Y plot and Y/Doppler plot
        end
        if P.guiMonitor(iGUI).logMagRange == 1
            TOTAL_PAYLOAD_SIZE_BYTES = TOTAL_PAYLOAD_SIZE_BYTES +...
                P.dataPath.numRangeBins(k) * 2 + tlSize;
            P.guiMonitor(iGUI).numFigures = P.guiMonitor(iGUI).numFigures + 1;
        end  
        if (P.guiMonitor(iGUI).noiseProfile == 1 &&...
                P.guiMonitor(iGUI).logMagRange == 1)
            TOTAL_PAYLOAD_SIZE_BYTES = TOTAL_PAYLOAD_SIZE_BYTES +...
                P.dataPath.numRangeBins(k) * 2 + tlSize;
        end      
        if (P.guiMonitor(iGUI).rangeAzimuthHeatMap == 1)
            TOTAL_PAYLOAD_SIZE_BYTES = TOTAL_PAYLOAD_SIZE_BYTES +...
                (P.dataPath.numTxAzimAnt * P.dataPath.numRxAnt) *...
                P.dataPath.numRangeBins(k) * 4 + tlSize;
            P.guiMonitor(iGUI).numFigures = P.guiMonitor(iGUI).numFigures + 1; 
        end
        if (P.guiMonitor(iGUI).rangeDopplerHeatMap == 1)
            TOTAL_PAYLOAD_SIZE_BYTES = TOTAL_PAYLOAD_SIZE_BYTES +...
                P.dataPath.numDopplerBins(k) * P.dataPath.numRangeBins(k) * 2 + tlSize;
            P.guiMonitor(iGUI).numFigures = P.guiMonitor(iGUI).numFigures + 1;
        end
        if (P.guiMonitor(iGUI).stats == 1)
            TOTAL_PAYLOAD_SIZE_BYTES = TOTAL_PAYLOAD_SIZE_BYTES +...
                STATS_SIZE_BYTES + tlSize;
            P.guiMonitor(iGUI).numFigures = P.guiMonitor(iGUI).numFigures + 1;
        end
        TOTAL_PAYLOAD_SIZE_BYTES = 32 * floor((TOTAL_PAYLOAD_SIZE_BYTES + 31) / 32);
        P.guiMonitor(iGUI).numFigRow = 2;
        P.guiMonitor(iGUI).numFigCol = ceil(P.guiMonitor(iGUI).numFigures /...
            P.guiMonitor(iGUI).numFigRow);
    
        if (platformType == hex2dec('a1642'))
            [P.dspFftScaleComp2D_lin, P.dspFftScaleComp2D_log] = dspFftScalComp2(16, P.dataPath.numDopplerBins(k));
            [P.dspFftScaleComp1D_lin, P.dspFftScaleComp1D_log]  = dspFftScalComp1(64, P.dataPath.numRangeBins(k));

        elseif (platformType == hex2dec('a1842'))
            [P.dspFftScaleComp1D_lin, P.dspFftScaleComp1D_log] = dspFftScalComp2(32, P.dataPath.numRangeBins(k));
            P.dspFftScaleComp2D_lin = 1;
            P.dspFftScaleComp2D_log = 0;
        else
            [P.dspFftScaleComp1D_lin, P.dspFftScaleComp1D_log] = dspFftScalComp2(32, P.dataPath.numRangeBins(k));
            P.dspFftScaleComp2D_lin = 1;
            P.dspFftScaleComp2D_log = 0;
        end
        P.dspFftScaleCompAll_lin(k) = P.dspFftScaleComp2D_lin * P.dspFftScaleComp1D_lin;
        P.dspFftScaleCompAll_log(k) = P.dspFftScaleComp2D_log + P.dspFftScaleComp1D_log;    
    end
    
    P.dataPath.numAzimuthBins = NUM_ANGLE_BINS;
    
    if strcmpi(gPlatform, 'xwr6843aop')
        antDef.tx.col = [0 0 2];
        antDef.tx.row = [0 2 2];
        antDef.rx.col =[1 1 0 0];
        antDef.rx.row =[0 1 0 1];

        P.dataPath.antDef.tx.col = antDef.tx.col(P.dataPath.txAntOrder+1); %+1 for one base
        P.dataPath.antDef.tx.row = antDef.tx.row(P.dataPath.txAntOrder+1);
        
        P.dataPath.antDef.rx.col = antDef.rx.col;
        P.dataPath.antDef.rx.row = antDef.rx.row;
        
        LUT = [];
        ind = 1;
        for txIdx = 1:P.dataPath.numTxAnt
            for rxIdx = 1:P.dataPath.numRxAnt
                LUT.row(ind) = P.dataPath.antDef.tx.row(txIdx) + P.dataPath.antDef.rx.row(rxIdx);
                LUT.col(ind) = P.dataPath.antDef.tx.col(txIdx) + P.dataPath.antDef.rx.col(rxIdx);
                ind = ind + 1;
            end
        end
        rowMax = max(LUT.row);
        rowMin = min(LUT.row);
        colMax = max(LUT.col);
        colMin = min(LUT.col);
%         LUT.row = LUT.row - rowMin;
%         LUT.col = LUT.col - colMin;
        P.dataPath.numRow = rowMax + 1;
        P.dataPath.numCol = colMax + 1;
        
        %Find backwar LUT for azimuth heatmap
        backwardLUT = [];
        lutIdx = 1;
        for colIdx = 0:P.dataPath.numCol-1
            for ind = 1: P.dataPath.numTxAnt * P.dataPath.numRxAnt
                if LUT.row(ind) == (P.dataPath.numRow-1) && LUT.col(ind) == colIdx %last row
                     backwardLUT(lutIdx) = ind; % one based
                     lutIdx = lutIdx + 1;
                     break;
                end
                
            end
        end
        P.dataPath.antBackwardMapLUT = backwardLUT;

        forwardLUT = [];
        for ind = 1: P.dataPath.numTxAnt * P.dataPath.numRxAnt
             forwardLUT(ind) = LUT.col(ind) * P.dataPath.numRow + LUT.row(ind) + 1; %+1 for one based
        end
        P.dataPath.antForwardMapLUT = forwardLUT;
        
    end
return

function [y] = pow2roundup (x)
    y = 1;
    while x > y
        y = y * 2;
    end
return

function myKeyPressFcn(hObject, event)
    global EXIT_KEY_PRESSED
    if lower(event.Key) == 'q'
        EXIT_KEY_PRESSED  = 1;
    end

return

function Resize_clbk(hObject, event, figHnd)
global Params
displayChirpParams(Params, figHnd);
return

function []=plotGrid(figHnd, R,range_grid_arc)

sect_width=pi/12;  
offset_angle=pi/6:sect_width:5*pi/6;
r=[0:range_grid_arc:R];
w=linspace(pi/6,5*pi/6,128);

for n=2:length(r)
    plot(figHnd, real(r(n)*exp(1j*w)),imag(r(n)*exp(1j*w)),'color',[0.5 0.5 0.5], 'linestyle', ':')
end


for n=1:length(offset_angle)
    plot(figHnd, real([0 R]*exp(1j*offset_angle(n))),imag([0 R]*exp(1j*offset_angle(n))),'color',[0.5 0.5 0.5], 'linestyle', ':')
end
return

function [Header, idx] = getHeader(bytevec, idx)
    idx = idx + 8; %Skip magic word
    Header.subFrameNumber = 0;
    word = [1 256 65536 16777216]';
    Header.version = sum(bytevec(idx+[1:4]) .* word);
    idx = idx + 4;
    Header.totalPacketLen = sum(bytevec(idx+[1:4]) .* word);
    idx = idx + 4;
    Header.platform = sum(bytevec(idx+[1:4]) .* word);
    idx = idx + 4;
    Header.frameNumber = sum(bytevec(idx+[1:4]) .* word);
    idx = idx + 4;
    Header.timeCpuCycles = sum(bytevec(idx+[1:4]) .* word);
    idx = idx + 4;
    Header.numDetectedObj = sum(bytevec(idx+[1:4]) .* word);
    idx = idx + 4;
    Header.numTLVs = sum(bytevec(idx+[1:4]) .* word);
    idx = idx + 4;
    if Header.platform == hex2dec('a1642')
        Header.subFrameNumber = sum(bytevec(idx+[1:4]) .* word);
        idx = idx + 4;
    end
    if Header.platform == hex2dec('a1842')
        Header.subFrameNumber = sum(bytevec(idx+[1:4]) .* word);
        idx = idx + 4;
    end
    if Header.platform == hex2dec('A1843') ||...
       Header.platform == hex2dec('A6843')     
        Header.subFrameNumber = sum(bytevec(idx+[1:4]) .* word);
        idx = idx + 4;
    end
return

function [tlv, idx] = getTlv(bytevec, idx)
    if (idx+(1:4)) > length(bytevec)
        fprintf ('Index exceeds the number of array elements, bytevec length = %d, idx = %d \n', length(bytevec), idx);
        tlv.type = -1;
        tlv.length = 0;
        idx = 0;
        return;
    end
    word = [1 256 65536 16777216]';
    tlv.type = sum(bytevec(idx+(1:4)) .* word);
    idx = idx + 4;
    tlv.length = sum(bytevec(idx+(1:4)) .* word);
    idx = idx + 4;
return

function [detObj, idx] = getDetObj(bytevec, idx, tlvLen, numObj)
    global OBJ_STRUCT_SIZE_BYTES;
    detObj =[];
    detObj.numObj = numObj;
    if tlvLen > 0
        %Get detected object descriptor
        %word = [1 256]';
        %detObj.numObj = sum(bytevec(idx+(1:2)) .* word);
        %idx = idx + 2;
        
        %Get detected array of detected objects
        bytes = bytevec(idx+(1:detObj.numObj*OBJ_STRUCT_SIZE_BYTES));
        idx = idx + detObj.numObj*OBJ_STRUCT_SIZE_BYTES;

        bytes = reshape(bytes, OBJ_STRUCT_SIZE_BYTES, detObj.numObj);
        ofs = 0;
        tmp = uint8(bytes(ofs + (1:4),:));
        detObj.x = typecast(tmp(:), 'single');
        ofs = ofs + 4;

        tmp = uint8(bytes(ofs + (1:4),:));
        detObj.y = typecast(tmp(:), 'single');
        ofs = ofs + 4;

        tmp = uint8(bytes(ofs + (1:4),:));
        detObj.z = typecast(tmp(:), 'single');
        ofs = ofs + 4;

        tmp = uint8(bytes(ofs + (1:4),:));
        detObj.doppler = typecast(tmp(:), 'single');
    end
return

function [sideInfo, idx] = getSideInfo(bytevec, idx, tlvLen, numObj)
    global SIDE_INFO_STRUCT_SIZE_BYTES;
    sideInfo =[];
    sideInfo.numObj = numObj;
    if tlvLen > 0
        %Get detected object descriptor
        %word = [1 256]';
        %sideInfo.numObj = sum(bytevec(idx+(1:2)) .* word);
        %idx = idx + 2;
        
        %Get detected array of detected objects
        bytes = bytevec(idx+(1:sideInfo.numObj*SIDE_INFO_STRUCT_SIZE_BYTES));
        idx = idx + sideInfo.numObj*SIDE_INFO_STRUCT_SIZE_BYTES;

        bytes = reshape(bytes, SIDE_INFO_STRUCT_SIZE_BYTES, sideInfo.numObj);
        ofs = 0;
        tmp = uint8(bytes(ofs + (1:2),:));
        sideInfo.snr = typecast(tmp(:), 'int16');
        ofs = ofs + 2;

        tmp = uint8(bytes(ofs + (1:2),:));
        sideInfo.noise = typecast(tmp(:), 'int16');
    end
return

function [rp, idx] = getRangeProfile(bytevec, idx, len, rangeProfile)
    rp = bytevec(idx+(1:len));
    idx = idx + len;
    rp=rp(1:2:end)+rp(2:2:end)*256;
    %workaround for range DPU limitation
    if length(rp) == 1022
       rp = [rp; rp(end-1:end)]; 
    end
    if rangeProfile == 0
     show2DAngleFft = 0;        
        if (show2DAngleFft)        
            y=rp(129+[0:31]); %Mapping output
            x=rp(129+32+[0:23]); %Mapping input
            x(x>32767) = x(x>32767)-65536;
            y(y>32767) = y(y>32767)-65536;
            rp(129+[0:31])=0;
            rp(129+32+[0:23])=0;
            x=x(1:2:end)+1j*x(2:2:end);
            y=y(1:2:end)+1j*y(2:2:end);
            numAntRow = 4;
            numAntCol = 4;
            y=reshape(y,numAntCol,numAntRow);
        
%         figure(100);
%         hold off
%         plot(y(:,1),'bo-');
%         hold on
%         plot(y(:,2),'ro-');
%         plot(y(3:4,3),'co-');
%         plot(y(3:4,4),'mo-');
%         title('Azimuth')
%         axis equal
%         axis square
%         axis(5000*[-1 1 -1 1])
%         
%         figure(101);
%         hold off
%         plot(y(3,:),'bo-');
%         hold on
%         plot(y(4,:),'ro-');
%         plot(y(3,3:4),'co-');
%         plot(y(4,3:4),'mo-');
%         title('Elevation')
%         axis equal
%         axis square
%         axis(5000*[-1 1 -1 1])

        
        %A=[0 0; 0 1;0 2;0 3;  1 0; 1 1;1 2;1 3;  2 0; 2 1;2 2;2 3;  3 0; 3 1;3 2;3 3;];
        %labels=cellstr(num2str(A,'(%d,%d)'));
        %text(real(y), imag(y), labels, 'VerticalAlignment','bottom', 'HorizontalAlignment','right')
%         axis equal
%         axis square
%         axis(5000*[-1 1 -1 1])
%         title('AFTER MAPPING')
        end
        if (0)%(show2DAngleFft)
            figure(101)
            y1=fft(y,64);
            y2=fft(y1,64,2);
            surf([-32:31],[-32:31],abs(fftshift(fftshift(y2.',1),2)))
            view(2)

            if  (numAntRow == 4) && (numAntRow == 4)
                figure(201)
                hold off
                plot(32*[-1 1 1  -1 -1],32*[-1 -1 1 1 -1])
                y1= fftshift(abs(fft(y(:,4),64)));
                y2= fftshift(abs(fft(y(1,:),64)));
                ymax=max(max(y1),max(y2));
                hold on
                plot([-32:31],32*y1/ymax+32)
                plot(32*y2/ymax+32,[32:-1:-31])
                [y1m,i1m]=max(y1);
                [y2m,i2m]=max(y2);
                plot([i1m i1m]-32,[-32 32])
                plot([-32 32],32-[i2m i2m]) 
                axis equal
                axis square
                axis(80*[-1 1 -1 1])
                title(sprintf('%d,%d',i1m, i2m))
            end
            if (0)%(numAntRow == 2) 
                figure(201)
                hold off
                plot(32*[-1 1 1  -1 -1],32*[-1 -1 1 1 -1])
                y1= fftshift(abs(fft(y(:,2),64)));
                y2= fftshift(abs(fft(y(4,:),64)));
                ymax=max(max(y1),max(y2))
                hold on
                plot([-32:31],32*y1/ymax+32)
                plot(32*y2/ymax+32,[32:-1:-31])
                [y1m,i1m]=max(y1);
                [y2m,i2m]=max(y2);
                plot([i1m i1m]-32,[-32 32])
                plot([-32 32],32-[i2m i2m]) 
                axis equal
                axis square
                axis(80*[-1 1 -1 1])
            end
        end

%         figure(102)
%         hold off
%         A=[0 0; 0 1;0 2;0 3;  1 0; 1 1;1 2;1 3;  2 0; 2 1;2 2;2 3;];
%         labels=cellstr(num2str(A,'(%d,%d)'));
%         plot(x,'.')
%         hold on
%         plot(0,0,'k+')
%         txAnt=[1:4];
%         text(real(x(1:4)), imag(x(1:4)), labels(1:4), 'VerticalAlignment','bottom', 'HorizontalAlignment','right','color','red')
%         txAnt=[5:8];
%         text(real(x(txAnt)), imag(x(txAnt)), labels(txAnt), 'VerticalAlignment','bottom', 'HorizontalAlignment','right','color','green')
%         txAnt=[9:12];
%         text(real(x(txAnt)), imag(x(txAnt)), labels(txAnt), 'VerticalAlignment','bottom', 'HorizontalAlignment','right','color','blue')
%         axis equal
%         axis square
%         axis(5000*[-1 1 -1 1])
%         title('BEFORE MAPPING')
        
        

    end
    
return

function [Q, Qcorr, q, idx] = getAzimuthStaticHeatMap(bytevec, idx, numTxAzimAnt, numRxAnt, numRangeBins, numAngleBins, nearEndCorr, tlvLen)
    workAround = 0;
    if tlvLen == (numTxAzimAnt * numRxAnt * (numRangeBins-2) * 4)
        workAround = 1;
        numRangeBins = numRangeBins - 2;
    end
    len = numTxAzimAnt * numRxAnt * numRangeBins * 4;
    q = bytevec(idx+(1:len));
    idx = idx + len;
    q = q(1:2:end)+q(2:2:end)*256;
    q(q>32767) = q(q>32767) - 65536;
    q = 1j*q(1:2:end)+q(2:2:end); %Imaginary first, real second
    q = reshape(q, numTxAzimAnt * numRxAnt, numRangeBins);
    
    if workAround
        q = [q q(:,end-1:end)];
    end
    
    %remove 3 range bins...
    %q=circshift(q,-3,2); %for 512
    %q=circshift(q,-2,2); %for 256
     

    Q = fft(q, numAngleBins);

    %Near field corrected
    q1=q;
    q2=q;
    q1(5:8,:) = 0;
    q2(1:4,:) = 0;
    Q1 = fft(q1, numAngleBins);
    Q2 = fft(q2, numAngleBins);
    Qcorr = Q1 + Q2.* nearEndCorr;

return

function plotAngleFFT(P, q, rngIdx)
    if rngIdx < 0
        return;
    end
    x=q(:,rngIdx);
    
    y=zeros(P.dataPath.numRow*P.dataPath.numCol,1);
    y(P.dataPath.antForwardMapLUT)=x;
    y=reshape(y, P.dataPath.numRow, P.dataPath.numCol);
    y1=fft(y,64);
    y2=fft(y1,64,2);
    
    surf(P.guiMonitor.figAzimuthHeatmap.hax(4),...
        [-32:31],...
        [-32:31],...
        abs(fftshift(fftshift(y2,1),2)));
    view(P.guiMonitor.figAzimuthHeatmap.hax(4), 2);
    xlabel(P.guiMonitor.figAzimuthHeatmap.hax(4), 'Azimuth index');
    ylabel(P.guiMonitor.figAzimuthHeatmap.hax(4), 'Elevation index');
    axis(P.guiMonitor.figAzimuthHeatmap.hax(4), 'square');
    shading(P.guiMonitor.figAzimuthHeatmap.hax(4), 'interp'); 
return

function [Q, q, idx] = getAzimElevStaticHeatMap(P, bytevec, idx, numTxAnt, numRxAnt, numRangeBins, numAngleBins, tlvLen)
    workAround = 0;
    if tlvLen == (numTxAnt * numRxAnt * (numRangeBins-2) * 4)
        workAround = 1;
        numRangeBins = numRangeBins - 2;
    end
    len = numTxAnt * numRxAnt * numRangeBins * 4;
    q = bytevec(idx+(1:len));
    idx = idx + len;
    q = q(1:2:end)+q(2:2:end)*256;
    q(q>32767) = q(q>32767) - 65536;
    q = 1j*q(1:2:end)+q(2:2:end); %Imaginary first, real second
    q = reshape(q, numTxAnt * numRxAnt, numRangeBins);
    
    qAzim = q(P.dataPath.antBackwardMapLUT,:);
    
    if workAround
        qAzim = [qAzim qAzim(:,end-1:end)];
    end
        
    %remove 3 range bins...
    %q=circshift(q,-3,2); %for 512
    %q=circshift(q,-2,2); %for 256
     

    Q = fft(qAzim, numAngleBins);
return

function [rangeDoppler, idx] = getRangeDopplerHeatMap(bytevec, idx, numDopplerBins, numRangeBins, tlvLen)
    workAround = 0;
    if tlvLen == numDopplerBins * (numRangeBins-2) * 2
        workAround = 1;
        numRangeBins = numRangeBins - 2;
    end
    len = numDopplerBins * numRangeBins * 2;
    rangeDoppler = bytevec(idx+(1:len));
    idx = idx + len;
    rangeDoppler = rangeDoppler(1:2:end) + rangeDoppler(2:2:end)*256;
    rangeDoppler = reshape(rangeDoppler, numDopplerBins, numRangeBins);
    rangeDoppler = fftshift(rangeDoppler,1);
    if workAround
        rangeDoppler = [rangeDoppler rangeDoppler(:,end-1:end)];
    end
return

function [StatsInfo, idx] = getStatsInfo(bytevec, idx)
    word = [1 256 65536 16777216]';
    StatsInfo.interFrameProcessingTime = sum(bytevec(idx+(1:4)) .* word);
    idx = idx + 4;
    StatsInfo.transmitOutputTime = sum(bytevec(idx+(1:4)) .* word);
    idx = idx + 4;
    StatsInfo.interFrameProcessingMargin = sum(bytevec(idx+(1:4)) .* word);
    idx = idx + 4;
    StatsInfo.interChirpProcessingMargin = sum(bytevec(idx+(1:4)) .* word);
    idx = idx + 4;
    StatsInfo.activeFrameCPULoad = sum(bytevec(idx+(1:4)) .* word);
    idx = idx + 4;
    StatsInfo.interFrameCPULoad = sum(bytevec(idx+(1:4)) .* word);
    idx = idx + 4;
return

function [sLin, sLog] = dspFftScalComp2(fftMinSize, fftSize)
    sLin = fftMinSize/fftSize;
    sLog = 20*log10(sLin);
return

function [sLin, sLog] = dspFftScalComp1(fftMinSize, fftSize)
    smin =  (2.^(ceil(log2(fftMinSize)./log2(4)-1)))  ./ (fftMinSize);
    sLin =  (2.^(ceil(log2(fftSize)./log2(4)-1)))  ./ (fftSize);
    sLin = sLin / smin;
    sLog = 20*log10(sLin);
return

% function [nearEndCorr] = nearEndCorrectionCalc(P)
%     numAzimuthBins = 64;
%     numRangeBins = P.dataPath.numRangeBins; 
%     rangeIdxToMeters = P.dataPath.rangeIdxToMeters;
%     lam = 3e8/( P.profileCfg.startFreq * 1e9 )*1000; % mm
% 
%     nearEndCorr = zeros(numAzimuthBins,numRangeBins);
%     % horizontal distance between tx1 and rx4, measured on the 16xx EVM to be 8.7 mm approx
%     h_tx1_rx4 = 8.7;
%     for rangeIdx = 0 : (numRangeBins-1)
%         Ran = rangeIdxToMeters * rangeIdx *1000 - 60;
%         
%         Ran = max(Ran, 0);
%         for azimIdx = 0 : (numAzimuthBins-1)
%             if azimIdx < numAzimuthBins/2
%                 azimIdxS = azimIdx;
%             else
%                 azimIdxS = azimIdx - numAzimuthBins;
%             end
%             th = asind(2*azimIdxS/numAzimuthBins);
% 
%             tx1 = 360/lam * sqrt(Ran^2 + lam^2 - 2*Ran*lam*cosd(90 - th));
%             rx4_tx1 = 360/lam * sqrt(Ran^2 + (lam + h_tx1_rx4)^2 - 2*Ran*(lam + h_tx1_rx4)*cosd(90 - th));
% 
%             tx2 = 360/lam * sqrt(Ran^2 + lam^2 - 2*Ran*lam*cosd(90 + th));
%             rx1_tx2 = 360/lam * sqrt(Ran^2 + (5*lam/2 + h_tx1_rx4)^2 - 2*Ran*(5*lam/2 + h_tx1_rx4)*cosd(90 - th));
% 
%             if Ran > 0
%                 nearEndCorr(azimIdx+1, rangeIdx+1) = exp(-1j *pi/180*((tx2 + rx1_tx2) - (rx4_tx1 + tx1) - 180*sind(th)));
%             else
%                 nearEndCorr(azimIdx+1, rangeIdx+1) = exp(-1j*0);
%             end
%         end
%     end
% return

% function [nearEndCorr] = nearEndCorrectionCalc(P)
%     numAzimuthBins = 64;
%     numRangeBins = P.dataPath.numRangeBins; 
%     rangeIdxToMeters = P.dataPath.rangeIdxToMeters;
%     lam = 3e8/( P.profileCfg.startFreq * 1e9 )*1000; % mm
% 
%     nearEndCorr = zeros(numAzimuthBins,numRangeBins);
%     % horizontal distance between tx1 and rx4, measured on the 16xx EVM to be 8.7 mm approx
%     h_tx1_rx4 = 8.7;
%     for rangeIdx = 0 : (numRangeBins-1)
%         Ran = rangeIdxToMeters * rangeIdx *1000 - 60;
%         
%         Ran = max(Ran, 0);
%         for azimIdx = 0 : (numAzimuthBins-1)
%             if azimIdx < numAzimuthBins/2
%                 azimIdxS = azimIdx;
%             else
%                 azimIdxS = azimIdx - numAzimuthBins;
%             end
%             th = 2*azimIdxS/numAzimuthBins;
% 
%             tx1 = sqrt(Ran^2 + lam^2 - 2*Ran*lam*th);
%             rx4_tx1 = sqrt(Ran^2 + (lam + h_tx1_rx4)^2 - 2*Ran*(lam + h_tx1_rx4)*th);
% 
%             tx2 = sqrt(Ran^2 + lam^2 + 2*Ran*lam*th);
%             rx1_tx2 = sqrt(Ran^2 + (5*lam/2 + h_tx1_rx4)^2 - 2*Ran*(5*lam/2 + h_tx1_rx4)*th);
% 
%             if Ran > 0
%                 nearEndCorr(azimIdx+1, rangeIdx+1) = exp(-1j * (2*pi/lam*((tx2 + rx1_tx2) - (rx4_tx1 + tx1)) - pi*th));            else
%                 nearEndCorr(azimIdx+1, rangeIdx+1) = exp(-1j*0);
%             end
%         end
%     end
% return

function [nearEndCorr] = nearEndCorrectionCalc(P, subFrameIdx)
global NUM_ANGLE_BINS
    sfi = subFrameIdx;
    if (P.isAdvanceSubFrm == 1)
        chirpStartIdx = P.subFrameCfg(sfi).chirpStartIdx;
        profileIdx = P.chirpCfg(chirpStartIdx+1).profileIdx;
    else
        profileIdx = 0;
    end
    
    numAzimuthBins = P.dataPath.numAzimuthBins;
    numRangeBins = P.dataPath.numRangeBins(sfi); 
    rangeIdxToMeters = P.dataPath.rangeIdxToMeters(sfi);
    lam = 3e8 / ( P.profileCfg(profileIdx+1).startFreq * 1e9 ) * 1000; % mm

    nearEndCorr = zeros(numAzimuthBins, numRangeBins);
    % horizontal distance between tx1 and rx4, measured on the 16xx EVM to be 8.7 mm approx
    h_tx1_rx4 = 8.7;
    A = 0;
    B = lam;%(2+3/4)*lam + h_tx1_rx4; %2.5*lam;
    C = 2 * lam;
    D = C + h_tx1_rx4;
    E = D + 3* lam/2;
    rangeBiasMiliMeters = 65;    
    for rangeIdx = 0 : (numRangeBins-1)
        Ran = rangeIdxToMeters * rangeIdx *1000 - rangeBiasMiliMeters;
        
        Ran = max(Ran, 0);
        for azimIdx = 0 : (numAzimuthBins-1)
            if azimIdx < numAzimuthBins/2
                azimIdxS = azimIdx;
            else
                azimIdxS = azimIdx - numAzimuthBins;
            end
            th = 2*azimIdxS/numAzimuthBins;
            tha(azimIdx+1) = th;
            tx1 = sqrt(Ran^2 + (C-B)^2 - 2*Ran*(C-B)*th);
            rx4_tx1 = sqrt(Ran^2 + (D-B)^2 - 2*Ran*(D-B)*th);

            tx2 = sqrt(Ran^2 + (A-B)^2 - 2*Ran*(A-B)*th);
            rx1_tx2 = sqrt(Ran^2 + (E-B)^2 - 2*Ran*(E-B)*th);

            if Ran > 0
                nearEndCorr(azimIdx+1, rangeIdx+1) = exp(-1j * (2*pi/lam*((tx2 + rx1_tx2) - (rx4_tx1 + tx1)) - pi*th));            
            else
                nearEndCorr(azimIdx+1, rangeIdx+1) = exp(-1j*0);
            end
        end
    end
return


function output_txt = rp_dcm_callback(obj,event_obj)
% Display the position of the data cursor
% obj          Currently not used (empty)
% event_obj    Handle to event object
% output_txt   Data cursor text string (string or cell array of strings).
global Params
global sfIdx

haxis = get(obj,'Parent');
pos = get(event_obj,'Position');

if haxis == Params.figRangeProfile.hax1
    Params.selectedRngIdx = 1 + round(pos(1)/Params.dataPath.rangeIdxToMeters(sfIdx));
    output_txt = {[sprintf('%.2f m, %.0f dB', pos(1), pos(2))]};
elseif haxis == Params.figCloudPoint.hax1
    output_txt = {['X: ',num2str(pos(1),4)],...
                  ['Y: ',num2str(pos(2),4)]};
    if length(pos) > 2
        output_txt{end+1} = ['Z: ',num2str(pos(3),4)];
    end
else
    output_txt = {['X: ',num2str(pos(1),4)],...
                  ['Y: ',num2str(pos(2),4)]};
    if length(pos) > 2
        output_txt{end+1} = ['Z: ',num2str(pos(3),4)];
    end
end
return