%
% Copyright (c) 2020 Texas Instruments Incorporated
%
% All rights reserved not granted herein.
% Limited License.
%
% Texas Instruments Incorporated grants a world-wide, royalty-free,
% non-exclusive license under copyrights and patents it now or hereafter
% owns or controls to make, have made, use, import, offer to sell and sell ("Utilize")
% this software subject to the terms herein.  With respect to the foregoing patent
% license, such license is granted  solely to the extent that any such patent is necessary
% to Utilize the software alone.  The patent license shall not apply to any combinations which
% include this software, other than combinations with devices manufactured by or for TI ("TI Devices").
% No hardware patent is licensed hereunder.
%
% Redistributions must preserve existing copyright notices and reproduce this license (including the
% above copyright notice and the disclaimer and (if applicable) source code license limitations below)
% in the documentation and/or other materials provided with the distribution
%
% Redistribution and use in binary form, without modification, are permitted provided that the following
% conditions are met:
%
%             * No reverse engineering, decompilation, or disassembly of this software is permitted with respect to any
%               software provided in binary form.
%             * any redistribution and use are licensed by TI for use only with TI Devices.
%             * Nothing shall obligate TI to provide you with source code for the software licensed and provided to you in object code.
%
% If software source code is provided to you, modification and redistribution of the source code are permitted
% provided that the following conditions are met:
%
%   * any redistribution and use of the source code, including any resulting derivative works, are licensed by
%     TI for use only with TI Devices.
%   * any redistribution and use of any object code compiled from the source code and any resulting derivative
%     works, are licensed by TI for use only with TI Devices.
%
% Neither the name of Texas Instruments Incorporated nor the names of its suppliers may be used to endorse or
% promote products derived from this software without specific prior written permission.
%
% DISCLAIMER.
%
% THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
% BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
% IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
% OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

clear, clc, close all;
delete(instrfind);

global car_bg;
global totNumRows;
global person;
global person2;
global empty1;
global empty2;
global bg_row;
global bg_col;
global topButtonNewValue;
global botButtonNewValue;
%global topbt;
global botbt;
 
%% Configuration Parameters
[controlSerialPort, dataSerialPort, chirpConfigurationFileName, liveDemoFlag, fileLoop] = configDialog();
mmwDemoCliPrompt = char('vodDemo:/>');

%% Read Chirp Configuration file
cliCfg = readCfg(chirpConfigurationFileName);
Params = parseCfg(cliCfg);

%% Calculate car2rowFlag, and VOD State Machine initialization
tracker = Params.tracker;

totNumRows = Params.totNumRows; 
for idx = 1:Params.numZones
    tracker(idx).state = 0;
    tracker(idx).freeze = 0;
    tracker(idx).numEntryCount = 0;
    tracker(idx).freezeSourceFlag = 0;
    tracker(idx).detect2freeCount = 0;    
    classStr.xVol{idx} = [];
    classStr.yVol{idx} = [];
    classStr.zVol{idx} = [];
    classStr.decision(idx) = 0;
end
classStr.snrAcc = zeros(Params.numZones, 1);
classStr.frameCount = zeros(Params.numZones, 1);
classStr.mode = Params.classParam.mode; 


%% obtain sensor parameters
sensor.rangeMax = Params.dataPath.rangeResolutionMeters*Params.dataPath.numRangeBins;
sensor.rangeMin = 1;
sensor.azimFoV = Params.sensorPosition.azimuthFov * 2 * pi/180;
sensor.elevFoV = Params.sensorPosition.elevationFov * 2 * pi/180;
sensor.yzRot = Params.sensorPosition.yzRot;
sensor.xyRot = Params.sensorPosition.xyRot;
sensor.xzRot = Params.sensorPosition.xzRot;
sensor.framePeriod = Params.frameCfg.framePeriodicity; %in ms
sensor.rangeResolution = Params.dataPath.rangeResolutionMeters;
sensor.maxRadialVelocity = Params.dataPath.dopplerResolutionMps*Params.frameCfg.numLoops/2;
sensor.radialVelocityResolution = Params.dataPath.dopplerResolutionMps;
sensor.azim = linspace(-sensor.azimFoV/2, sensor.azimFoV/2, 128);
sensor.elev = linspace(-sensor.elevFoV/2, sensor.elevFoV/2, 128);
sensor.height = Params.sensorPosition.zOffset;

% Set the display (scene) size to bound the defined cuboids and sensor height
[Params, scene] = calcZoneAndSceneBoundary(Params);

% Desktop setup

figHandle = figure('Name', 'Visualizer','tag','mainFigure');
clf(figHandle);
set(figHandle, 'WindowStyle','normal');
set(figHandle,'Name','Texas Instruments - Overhead 3D VOD V1.0','NumberTitle','off')    
set(figHandle,'currentchar',' ')         % set a dummy character
warning off MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame
jframe=get(figHandle,'javaframe');
set(figHandle, 'MenuBar', 'none');
set(figHandle, 'Color', [0 0 0]);

scrSize = get (0, 'ScreenSize');
figx = scrSize(3) * 0.50;
figy = scrSize(4) - 80;

pause(0.1);
%set(jframe,'Maximized',1); 
set(gcf,'Position', [0 80 figx figy]);
pause(0.1);

countPause = 0;
countState = 0;
countFrames = 0;
zCount = zeros(1, Params.numZones);

figureTitles = {'Statistics', 'Occupancy', 'Point Cloud', 'Chirp Configuration', 'Control'};
figureGroup = [1, 2, 3, 4, 5];
numFigures = size(figureTitles, 2);
hFigure = zeros(1,numFigures);

% Setup tab dimensions
% |1| |2| |3|
% |4| | | | |
% |5| | | | |

% Tab 1: Statistics and Chirp Configuration, [left, bottom, width, height]
hTabGroup(1) = uitabgroup(figHandle, 'Position', [0.0 0.66 0.34 0.34]);
% Tab 2: Left Pane
%hTabGroup(2) = uitabgroup(figHandle, 'Position', [0.2 0.0 0.4 1]);
hTabGroup(2) = uitabgroup(figHandle, 'Position', [0.34 0.5 0.66 0.5]);
% Tab 3: Right Pane
%hTabGroup(3) = uitabgroup(figHandle, 'Position', [0.6 0.0 0.4 1]);
hTabGroup(3) = uitabgroup(figHandle, 'Position', [0.34 0.0 0.66 0.5]);
% Tab 4: Chirp Configuration
hTabGroup(4) = uitabgroup(figHandle, 'Position', [0.0 0.33 0.34 0.33]);
% Tab 5: Control
hTabGroup(5) = uitabgroup(figHandle, 'Position', [0.0 0.0 0.34 0.33]);

hStatGlobal = [];

for iFig = 1:numFigures
    hFigure(iFig) = uitab(hTabGroup(figureGroup(iFig)), 'Title', figureTitles{iFig});
    ax = axes('parent', hFigure(iFig));
    gca = ax;  

    if(strcmp(figureTitles{iFig},'Chirp Configuration'))
        set(gca, 'visible', 'off');
        pause(0.1);
        tablePosition = [0.1 0.05 0.9 0.9];
        h = displayChirpParams(Params, tablePosition,  hFigure(iFig));
        h.InnerPosition = [h.InnerPosition(1:2) h.Extent(3:4)];
    end
    
    if(strcmp(figureTitles{iFig},'Statistics'))     
        set(gca, 'visible', 'off');
        hStatGlobal(1) = text(0, 0.90, 'Frame #0', 'FontSize',11);
        hStatGlobal(2) = text(0, 0.75, 'Zone:  State:  Freeze:  Points:  SNR:', 'FontSize',11);
        for idx = 1:Params.numZones
            hStatGlobal(2+idx) = text(0, 0.75 - idx * 0.08, '        ','FontSize',11);
        end        
    end
    
    if(strcmp(figureTitles{iFig},'Control'))
        set(gca, 'visible', 'off');
        
        cFig = iFig;
        hRbRecord = uicontrol(hFigure(cFig),'Style','radio','String','Record','FontSize', 10,...
            'Units', 'normalized', 'Position',[0.1 0.1 0.3 0.1],'Value',0);
        hRbCount = uicontrol(hFigure(cFig),'Style','radio','String','Count','FontSize', 10,...
            'Units', 'normalized', 'Position',[0.4 0.20 0.3 0.1],'Value',0);
        hRbPause = uicontrol(hFigure(cFig),'Style','radio','String','Pause Count','FontSize', 10,...
            'Units', 'normalized', 'Position',[0.4 0.1 0.3 0.1],'Value',0);
        hPbExit = uicontrol(hFigure(cFig),'Style', 'pushbutton', 'String', 'Exit','FontSize', 10,...
            'Units', 'normalized','Position', [0.7 0.1 0.2 0.1],'Callback', @exitPressFcn);
        setappdata(hPbExit, 'exitKeyPressed', 0);

        % Create radio buttons in the right button group.
        bg_bottom = uibuttongroup(hFigure(cFig), 'Visible','on',... 
                           'Title', "Point Cloud Pane", 'Position', [0.1 0.4 .7 .6],...
                           'SelectionChangedFcn', @bg_bottom_selection);

        botbt(1) = uicontrol(bg_bottom,'Style','radiobutton',...
                           'String','Pause Display',...
                           'Position',[15 100 150 20],...
                           'HandleVisibility','on');

        botbt(2) = uicontrol(bg_bottom,'Style','radiobutton',...
                           'String','2D Point Cloud',...
                           'Position',[15 80 150 20],...
                           'HandleVisibility','on');

        botbt(3) = uicontrol(bg_bottom,'Style','radiobutton',...
                           'String','3D Zones Row 1',...
                           'Position',[15 60 150 20],...
                           'HandleVisibility','on');

        botbt(4) = uicontrol(bg_bottom,'Style','radiobutton',...
                           'String','3D Zones Row 2',...
                           'Position',[15 40 150 20],...
                           'HandleVisibility','on');
       
       if (totNumRows < 3)
           botbt(5) = uicontrol(bg_bottom,'Style','radiobutton',...
                           'String','Avg Zone SNR',...
                           'Position',[15 20 150 20],...
                           'HandleVisibility','on');
       else
           
           botbt(5) = uicontrol(bg_bottom,'Style','radiobutton',...
                           'String','3D Zones Row 3',...
                           'Position',[15 20 150 20],...
                           'HandleVisibility','on');
           botbt(6) = uicontrol(bg_bottom,'Style','radiobutton',...
                           'String','Avg Zone SNR',...
                           'Position',[15 0 150 20],...
                           'HandleVisibility','on');
           
       end

       set(bg_bottom,'SelectedObject',botbt(2));  % Set the default value
    end
end


car_xscale = 0.5;
xstart = (1.0 - car_xscale) / 2;
seat_pos(1,:) = [(0.23*car_xscale)+xstart 0.54 0.13*car_xscale 0.17]; %front row seat positions in the graphic
seat_pos(2,:) = [(0.66*car_xscale)+xstart 0.54 0.13*car_xscale 0.17];
seat_pos(3,:) = [(0.20*car_xscale)+xstart 0.25 0.13*car_xscale 0.17]; %second row seat positions in the graphic
seat_pos(4,:) = [(0.45*car_xscale)+xstart 0.25 0.13*car_xscale 0.17];
seat_pos(5,:) = [(0.70*car_xscale)+xstart 0.25 0.13*car_xscale 0.17];
seat_pos(6,:) = [(0.20*car_xscale)+xstart 0.05 0.13*car_xscale 0.17]; %third row seat positions in the graphic
seat_pos(7,:) = [(0.45*car_xscale)+xstart 0.05 0.13*car_xscale 0.17];
seat_pos(8,:) = [(0.70*car_xscale)+xstart 0.05 0.13*car_xscale 0.17];

initialize_occupancy_display(hFigure(2), Params, car_xscale);

topButtonOldValue = 2;
topButtonNewValue = 2;
botButtonOldValue = 2;
botButtonNewValue = 2;


fileFrameSize = 1000;

frameStatStruct = struct('targetFrameNum', [], 'bytes', [], 'numInputPoints', 0, 'numOutputPoints', 0, 'timestamp', 0, 'start', 0, 'benchmarks', zeros(10,1), 'done', 0, ...
    'pointCloud', [], 'targetList', [], 'indexArray', [], 'zoneMap',[]);
fHist = repmat(frameStatStruct, 1, fileFrameSize);

if(liveDemoFlag == true)
    %Configure data UART port with input buffer to hold 100+ frames
    hDataSerialPort = configureDataSport(dataSerialPort, 65536);
    
    %Send Configuration Parameters to AWR68xx
    hControlSerialPort = configureControlPort(controlSerialPort);
    timeOut = get(hControlSerialPort,'Timeout');
    set(hControlSerialPort,'Timeout',1);

    %Send CLI configuration to AWR68xx
    fprintf('Sending configuration from %s file to AWR68xx ...\n', chirpConfigurationFileName);
    for k=1:length(cliCfg)
        if (length(cliCfg{k}) > 1)

           len = strlength(cliCfg{k});
           if (len > 14)
              len = 14; %length of interiorBounds
           end

           sstr = extractBetween(cliCfg{k}, 1, len); % do not send interiorBounds commands to target
           if strcmp(sstr,'interiorBounds')
               continue;
           end

           fprintf(hControlSerialPort, cliCfg{k});
           fprintf('%s\n', cliCfg{k});
           echo = fgetl(hControlSerialPort); % Get an echo of a command
           done = fgetl(hControlSerialPort); % Get "Done"
           prompt = fread(hControlSerialPort, size(mmwDemoCliPrompt,2)); % Get the prompt back
        end
    end
    fclose(hControlSerialPort);
    delete(hControlSerialPort);

    syncPatternUINT64 = typecast(uint16([hex2dec('0102'),hex2dec('0304'),hex2dec('0506'),hex2dec('0708')]),'uint64');
    syncPatternUINT8  = typecast(uint16([hex2dec('0102'),hex2dec('0304'),hex2dec('0506'),hex2dec('0708')]),'uint8');

    frameHeaderStructType = struct(...
        'sync',             {'uint64', 8}, ... % See syncPatternUINT64 below
        'version',          {'uint32', 4}, ...
        'packetLength',     {'uint32', 4}, ... % In bytes, including header
        'platform',         {'uint32', 4}, ...
        'frameNumber',      {'uint32', 4}, ... % Starting from 1
        'subframeNumber',   {'uint32', 4}, ...
        'chirpProcessingMargin',        {'uint32', 4}, ... % 600MHz clocks
        'frameProcessingTimeInUsec',    {'uint32', 4}, ... % 600MHz clocks
        'trackingProcessingTimeInUsec', {'uint32', 4}, ... % 200MHz clocks
        'uartSendingTimeInUsec',        {'uint32', 4}, ... % 200MHz clocks
        'numTLVs' ,         {'uint16', 2}, ... % Number of TLVs in this frame
        'checksum',         {'uint16', 2});    % Header checksum

    tlvHeaderStruct = struct(...
        'type',             {'uint32', 4}, ... % TLV object Type
        'length',           {'uint32', 4});    % TLV object Length, in bytes, including TLV header

    % Point Cloud TLV reporting unit for all reported points
    pointUintStruct = struct(...
        'elevUnit',             {'float', 4}, ... % elevation, in rad
        'azimUnit',             {'float', 4}, ... % azimuth, in rad
        'dopplerUnit',          {'float', 4}, ... % Doplper, in m/s
        'rangeUnit',            {'float', 4}, ... % Range, in m
        'snrUnit',              {'float', 4});    % SNR, ratio

    % Point Cloud TLV object consists of an array of points.
    % Each point has a structure defined below
    pointStruct = struct(...
        'elevation',        {'int8', 1}, ... % elevation, in rad
        'azimuth',          {'int8', 1}, ... % azimuth, in rad
        'doppler',          {'int16', 2}, ... % Doplper, in m/s
        'range',            {'uint16', 2}, ... % Range, in m
        'snr',              {'uint16', 2});    % SNR, ratio

    frameHeaderLengthInBytes = lengthFromStruct(frameHeaderStructType);
    tlvHeaderLengthInBytes = lengthFromStruct(tlvHeaderStruct);
    pointLengthInBytes = lengthFromStruct(pointStruct);
    pointUnitLengthInBytes = lengthFromStruct(pointUintStruct);
    indexLengthInBytes = 1;

end
exitRequest = 0;
lostSync = 0;
gotHeader = 0;
outOfSyncBytes = 0;
runningSlow = 0;
bytesAvailableMax = 0;
dspLoadMax = 0;
uartTxMax = 0;

hPlotCloudHandleAll = [];
hPlotPoints3D = [];

skipProcessing = 0;
frameNum = 1;
targetFrameNum = 1;
frameNumLogged = 1;
recordingEnabled = 0;

ZoneCountOccupied = 0;

pointCloud = single(zeros(5,0));
point3D = single(zeros(3,0));
count1 = 0;
count2 = 0;

if(liveDemoFlag == false)
    matlabFileName = ['fHistRT_', num2str(fileLoop, '%04d'), '.mat'];
    if(isfile(matlabFileName))
        load(matlabFileName,'fHist');
        disp(['Loading data from ', matlabFileName, ' ...']);            
    else
        disp('Exiting');
        return;
    end
end

while(1)
    while (lostSync == 0)
        prevRecord = recordingEnabled;
        recordingEnabled = get(hRbRecord, 'Value');
        if (prevRecord == 0) && (recordingEnabled == 1) %user just enabled recording
            frameNum = 1;
        end

        if (liveDemoFlag == true)
            frameStart = tic;
            fHist(frameNum).timestamp = frameStart;
            bytesAvailable = get(hDataSerialPort,'BytesAvailable');            

            while(bytesAvailable < frameHeaderLengthInBytes)
                pause(0.01);
                count1 = count1 + 1;
                bytesAvailable = get(hDataSerialPort,'BytesAvailable');        
            end

            if(bytesAvailable > bytesAvailableMax)
                bytesAvailableMax = bytesAvailable;
            end
            fHist(frameNum).bytesAvailable = bytesAvailable;
            if(gotHeader == 0)
                %Read the header first
                [rxHeader, byteCount] = fread(hDataSerialPort, frameHeaderLengthInBytes, 'uint8');
            end

            if(byteCount ~= frameHeaderLengthInBytes)
                reason = 'Header Size is wrong';
                lostSync = 1;
                break;
            end

            bytesAvailable = bytesAvailable - byteCount;

            fHist(frameNum).start = 1000*toc(frameStart);

            magicBytes = typecast(uint8(rxHeader(1:8)), 'uint64');
            if(magicBytes ~= syncPatternUINT64)
                reason = 'No SYNC pattern';
                lostSync = 1;
                [rxDataDebug, byteCountDebug] = fread(hDataSerialPort, bytesAvailable - frameHeaderLengthInBytes, 'uint8');            
                break;
            end       
            if(validateChecksum(rxHeader) ~= 0)
                reason = 'Header Checksum is wrong';
                lostSync = 1;
                break; 
            end

            frameHeader = readToStruct(frameHeaderStructType, rxHeader);

            if(gotHeader == 1)
                if(frameHeader.frameNumber > targetFrameNum)
                    targetFrameNum = frameHeader.frameNumber;
                    disp(['Found sync at frame ',num2str(targetFrameNum),'(',num2str(frameNum),'), after ', num2str(1000*toc(lostSyncTime),3), 'ms']);
                    gotHeader = 0;
                else
                    reason = 'Old Frame';
                    gotHeader = 0;
                    lostSync = 1;
                    break;
                end
            end

            % We have a valid header
            targetFrameNum = frameHeader.frameNumber;
            dataLength = frameHeader.packetLength - frameHeaderLengthInBytes;

            if (recordingEnabled == 1)
                fHist(frameNum).targetFrameNum = frameNum; %targetFrameNum;
                fHist(frameNum).header = frameHeader;
                fHist(frameNum).bytes = dataLength; 
            end

            numInputPoints = 0;
            mIndex = [];

            if (dataLength > 0)
                while(bytesAvailable < dataLength)
                    pause(0.01);
                    count2 = count2 + 1;
                    bytesAvailable = get(hDataSerialPort,'BytesAvailable');
                end

                %Read all packet
                [rxData, byteCount] = fread(hDataSerialPort, double(dataLength), 'uint8');
                if(byteCount ~= double(dataLength))
                    reason = 'Data Size is wrong'; 
                    lostSync = 1;
                    break;  
                end
                offset = 0;

                fHist(frameNum).benchmarks(1) = 1000*toc(frameStart);

                % TLV Parsing
                for nTlv = 1:frameHeader.numTLVs
                    tlvType = typecast(uint8(rxData(offset+1:offset+4)), 'uint32');
                    tlvLength = typecast(uint8(rxData(offset+5:offset+8)), 'uint32');

                    if (tlvLength + offset > dataLength)
                        reason = 'TLV Size is wrong';
                        lostSync = 1;
                        break;                    
                    end
                    offset = offset + tlvHeaderLengthInBytes;
                    valueLength = tlvLength - tlvHeaderLengthInBytes;

                    switch(tlvType)
                        case 6
                            % Point Cloud TLV
                            % Get the unit scale for each point cloud dimension
                            pointUnit = typecast(uint8(rxData(offset+1: offset+pointUnitLengthInBytes)),'single');
                            elevUnit = pointUnit(1);
                            azimUnit = pointUnit(2);
                            dopplerUnit = pointUnit(3);
                            rangeUnit = pointUnit(4);
                            snrUnit = pointUnit(5);

                            offset = offset + pointUnitLengthInBytes;
                            numInputPoints = (valueLength - pointUnitLengthInBytes)/pointLengthInBytes;

                            if (numInputPoints > 0)    
                                % Get Point Cloud from the sensor
                                pointCloudTemp = typecast(uint8(rxData(offset+1: offset+valueLength- pointUnitLengthInBytes)),'uint8');

                                rangeInfo = (double(pointCloudTemp(6:pointLengthInBytes:end)) * 256 + double(pointCloudTemp(5:pointLengthInBytes:end)));
                                rangeInfo = rangeInfo * rangeUnit;

                                azimuthInfo =  double(pointCloudTemp(2:pointLengthInBytes:end));
                                indx = find(azimuthInfo >= 128);
                                azimuthInfo(indx) = azimuthInfo(indx) - 256;
                                azimuthInfo = azimuthInfo * azimUnit; % * pi/180;

                                elevationInfo =  double(pointCloudTemp(1:pointLengthInBytes:end));
                                indx = find(elevationInfo >= 128);
                                elevationInfo(indx) = elevationInfo(indx) - 256;
                                elevationInfo = elevationInfo * elevUnit; % * pi/180;

                            %    dopplerInfo = double(pointCloudTemp(4:pointLengthInBytes:end)) * 256 + double(pointCloudTemp(3:pointLengthInBytes:end));
                            %    indx = find(dopplerInfo >= 32768);
                            %    dopplerInfo(indx) = dopplerInfo(indx) - 65536;
                            %    dopplerInfo = dopplerInfo * dopplerUnit;

                                snrInfo = double(pointCloudTemp(8:pointLengthInBytes:end)) * 256 + double(pointCloudTemp(7:pointLengthInBytes:end));
                                snrInfo = snrInfo * snrUnit;
                                snrInfo(snrInfo<=1) = 1.1;

                                idx = 1:length(rangeInfo);

                                range = rangeInfo(idx)';
                                azim = azimuthInfo(idx)';
                                elev = elevationInfo(idx)';
           
                                doppler = zeros(1,numInputPoints);
                                snr = snrInfo(idx)';
                                pointCloud = [range; azim; elev; doppler; snr];
                                point3D = transformation(pointCloud, Params.sensorPosition);

                            end
                            offset = offset + valueLength  - pointUnitLengthInBytes;

                        otherwise
                            reason = 'TLV Type is wrong';
                            lostSync = 1;
                            break;
                    end
                end
            end
            fHist(frameNum).benchmarks(2) = 1000*toc(frameStart);

            if (numInputPoints == 0)
                pointCloud = single(zeros(5,0));
                point3D = single(zeros(4,0));
            end

            numOutputPoints = size(pointCloud,2);

            % Store Point cloud
            if (recordingEnabled == 1)
                fHist(frameNum).numInputPoints = numInputPoints;
                fHist(frameNum).numOutputPoints = numOutputPoints;    
                fHist(frameNum).pointCloud = pointCloud;
                fHist(frameNum).indexArray = mIndex;
                fHist(frameNum).benchmarks(3) = 1000*toc(frameStart);
            end

        else
            % liveDemoFlag = false
            if (frameNum == 100)
                disp(frameNum);
            end
            
            targetFrameNum = fHist(frameNum).targetFrameNum;
            frameHeader = fHist(frameNum).header;

            bytesAvailable = fHist(frameNum).bytesAvailable;
            numOutputPoints = fHist(frameNum).numOutputPoints;

            pointCloud = fHist(frameNum).pointCloud;
            point3D = transformation(pointCloud, Params.sensorPosition);            
        end

        if (get(hRbCount, 'Value') == 1)
            if (countState == 0) %init counts for counting
                countFrames = 0;
                countState = 1;
                zCount = zeros(1, Params.numZones);
            end

            countPause = get(hRbPause, 'Value');
        else
            countState = 0;
            countPause = 0;
        end        


        %-------------------------------------------------------
        % Process and Plot pointCloud data
        %-------------------------------------------------------

        % Assign (map) detected points to the defined zone cuboids
        [zoneMap, xLoc, yLoc, zLoc] = zone_assign(numOutputPoints, point3D, Params.numZones, Params.zone);

        % Run the VOD state machine to yield occupancy status
        [ZoneCountOccupied, tracker] = occupancyDetection(Params.numZones, tracker, point3D, zoneMap, Params.stateMach);
        
        % classification
        if (classStr.mode > 0)
           classStr = classification_logic_perSeat(Params, classStr, point3D, zoneMap, tracker);  
        end
        
        % Display the top display panel
        if (topButtonNewValue == 2)
            car_display(hFigure(2), tracker, classStr, totNumRows, car_xscale, seat_pos, frameNum);
        end
                  
        % Display the bottom display panel
        if totNumRows < 3
            switch botButtonNewValue            
                case 2, incabin_detectionDisplay(hFigure(3), xLoc, yLoc, zLoc, tracker, Params.zone, Params.intBound);
                case 3, zone_display(hFigure(3), Params.zone, xLoc, yLoc, zLoc, 1);
                case 4, zone_display(hFigure(3), Params.zone, xLoc, yLoc, zLoc, 2);
                case 5, snr_display(hFigure(3), tracker, Params.numZones, Params.stateMach{1}.overloadThreshold);
                otherwise
            end
        else 
            switch botButtonNewValue            
                case 2, incabin_detectionDisplay(hFigure(3), xLoc, yLoc, zLoc, tracker, Params.zone, Params.intBound);
                case 3, zone_display(hFigure(3), Params.zone, xLoc, yLoc, zLoc, 1);
                case 4, zone_display(hFigure(3), Params.zone, xLoc, yLoc, zLoc, 2);
                case 5, zone_display(hFigure(3), Params.zone, xLoc, yLoc, zLoc, 3);
                case 6, snr_display(hFigure(3), tracker, Params.numZones, Params.stateMach{1}.overloadThreshold);
                otherwise
            end
        end
        if (countState == 1) %update zone counts if counting
            countFrames = countFrames + 1;
            for idx = 1:Params.numZones
                zCount(idx) = zCount(idx) + tracker(idx).state;
            end
        end

        if (liveDemoFlag == true)
            fHist(frameNum).benchmarks(4) = 1000*toc(frameStart);        
        end

        %ind = mIndex < 200;

        string{1} = sprintf('Frame #%d, (%d)', frameNum, targetFrameNum);

        if (countState == 1)
            if (countPause == 0)
                string{2} = sprintf('Zone:  Occupied:   Frames:   Count:');
                for idx = 1:Params.numZones
                   string{2+idx} = sprintf('     %d          %d          %d          %d', idx, tracker(idx).state, countFrames, zCount(idx));
                end                 
            end
        else
            string{2} = sprintf('Zone:  Occupied:   Points:   SNR:');
            for idx = 1:Params.numZones
               string{2+idx} = sprintf('    %d          %d           %2d        %4.1f', idx, tracker(idx).state, tracker(idx).numPoints, tracker(idx).avgSnr);
            end                 
        end

        set(hStatGlobal(1),'String',string{1});
        set(hStatGlobal(2),'String',string{2});

        for idx = 1:Params.numZones
            if (tracker(idx).freeze == 0)
                set(hStatGlobal(2+idx),'String',string{2+idx}, 'Color', 'Black');
            else
                set(hStatGlobal(2+idx),'String',string{2+idx}, 'Color', 'Red');
            end
        end

        if (getappdata(hPbExit, 'exitKeyPressed') == 1) 
            if (recordingEnabled)
                matlabFileName = ['fHistRT_', num2str(fileLoop, '%04d'), '.mat'];
                fHist = fHist(1:frameNum);
                save(matlabFileName,'fHist');        
                disp(['Saving data in ', matlabFileName, ' ...']);
            end
            disp('Exiting');
            close all;
            return;
        end
        
        frameNum = frameNum + 1;

        if (frameNum > fileFrameSize)
            if (liveDemoFlag == true)               
                if (recordingEnabled == 1)
                    matlabFileName = ['fHistRT_', num2str(fileLoop, '%04d'), '.mat'];
                    save(matlabFileName,'fHist');
                    disp(['Saving data in ', matlabFileName, ' ...']);
                    fHist = repmat(frameStatStruct, 1, fileFrameSize);
                end
            else
                matlabFileName = ['fHistRT_', num2str(fileLoop + 1, '%04d'), '.mat'];
                if(isfile(matlabFileName))
                    load(matlabFileName,'fHist');
                    disp(['Loading data from ', matlabFileName, ' ...']);
                else
                    disp('Exiting');
                    return;                    
                end
            end
            frameNum = 1;
            fileLoop = fileLoop + 1;
        end       
        
        if ((liveDemoFlag == false) && (frameNum == length(fHist)))
            return;
        end
       
        if(bytesAvailable > 32000)
            runningSlow  = 1;
        elseif(bytesAvailable < 1000)
            runningSlow = 0;
        end
        
        if(runningSlow)
            % Don't pause, we are slow
        else
            pause(0.02);
        end
    end
    
    lostSyncTime = tic;
    bytesAvailable = get(hDataSerialPort,'BytesAvailable');
    disp(['Lost sync at frame ', num2str(targetFrameNum),'(', num2str(frameNum), '), Reason: ', reason, ', ', num2str(bytesAvailable), ' bytes in Rx buffer']);

%{
    % To catch up, we read and discard all uart data
    bytesAvailable = get(hDataSerialPort,'BytesAvailable');
    disp(bytesAvailable);
    [rxDataDebug, byteCountDebug] = fread(hDataSerialPort, bytesAvailable, 'uint8');
%}    
    while(lostSync)
        for n=1:8
            [rxByte, byteCount] = fread(hDataSerialPort, 1, 'uint8');
            if(rxByte ~= syncPatternUINT8(n))
                outOfSyncBytes = outOfSyncBytes + 1;
                break;
            end
        end
        if(n == 8)
            lostSync = 0;
            frameNum = frameNum + 1;
            if(frameNum > 10000)
                frameNum = 1;
            end
            
            [header, byteCount] = fread(hDataSerialPort, frameHeaderLengthInBytes - 8, 'uint8');
            rxHeader = [syncPatternUINT8'; header];
            byteCount = byteCount + 8;
            gotHeader = 1;
        end
    end
end
disp('Done');
delete(instrfind);


%Display Chirp parameters in table on screen
function h = displayChirpParams(Params, Position, Parent)

    dat =  {'Start Frequency (Ghz)', Params.profileCfg.startFreq;...
            'Slope (MHz/us)', Params.profileCfg.freqSlopeConst;...   
            'Samples per chirp', Params.profileCfg.numAdcSamples;...
            'Chirps per frame',  Params.dataPath.numChirpsPerFrame;...
            'Frame duration (ms)',  Params.frameCfg.framePeriodicity;...
            'Sampling rate (Msps)', Params.profileCfg.digOutSampleRate / 1000;...
            'Bandwidth (GHz)', Params.profileCfg.freqSlopeConst * Params.profileCfg.numAdcSamples /...
                               Params.profileCfg.digOutSampleRate;...
            'Range resolution (m)', Params.dataPath.rangeResolutionMeters;...
            'Velocity resolution (m/s)', Params.dataPath.dopplerResolutionMps;...
            'Number of Rx (MIMO)', Params.dataPath.numRxAnt; ...
            'Number of Tx (MIMO)', Params.dataPath.numTxAnt;};
    columnname =   {'Chirp Parameter (Units)', 'Value'};
    columnformat = {'char', 'numeric'};
    
    h = uitable('Units','normalized', ...
            'Parent', Parent, ...
            'Position', Position, ...
            'Data', dat,... 
            'ColumnName', columnname,...
            'ColumnFormat', columnformat,...
            'ColumnWidth', 'auto',...
            'RowName',[]);
end

function [P] = parseCfg(cliCfg)
    P=[];
    for k=1:length(cliCfg)
        C = strsplit(cliCfg{k});
        if strcmp(C{1},'channelCfg')
            P.channelCfg.txChannelEn = str2double(C{3});
            P.dataPath.numTxAzimAnt = bitand(bitshift(P.channelCfg.txChannelEn,0),1) +...
            bitand(bitshift(P.channelCfg.txChannelEn,-1),1) + ...
            bitand(bitshift(P.channelCfg.txChannelEn,-2),1);
            P.dataPath.numTxElevAnt = 0;
            P.channelCfg.rxChannelEn = str2double(C{2});
            P.dataPath.numRxAnt = bitand(bitshift(P.channelCfg.rxChannelEn,0),1) +...
                                  bitand(bitshift(P.channelCfg.rxChannelEn,-1),1) +...
                                  bitand(bitshift(P.channelCfg.rxChannelEn,-2),1) +...
                                  bitand(bitshift(P.channelCfg.rxChannelEn,-3),1);
            P.dataPath.numTxAnt = P.dataPath.numTxElevAnt + P.dataPath.numTxAzimAnt;
                                
        elseif strcmp(C{1},'dataFmt')
        elseif strcmp(C{1},'profileCfg')
            P.profileCfg.startFreq = str2double(C{3});
            P.profileCfg.idleTime =  str2double(C{4});
            P.profileCfg.rampEndTime = str2double(C{6});
            P.profileCfg.freqSlopeConst = str2double(C{9});
            P.profileCfg.numAdcSamples = str2double(C{11});
            P.profileCfg.digOutSampleRate = str2double(C{12}); %uints: ksps
        elseif strcmp(C{1},'chirpCfg')
        elseif strcmp(C{1},'frameCfg')
            P.frameCfg.chirpStartIdx = str2double(C{2});
            P.frameCfg.chirpEndIdx = str2double(C{3});
            P.frameCfg.numLoops = str2double(C{4});
            P.frameCfg.numFrames = str2double(C{5});
            P.frameCfg.framePeriodicity = str2double(C{6});
        elseif strcmp(C{1},'guiMonitor')
            P.guiMonitor.detectedObjects = str2double(C{2});
            P.guiMonitor.logMagRange = str2double(C{3});
            P.guiMonitor.rangeAzimuthHeatMap = str2double(C{4});
            P.guiMonitor.rangeDopplerHeatMap = str2double(C{5});
        elseif strcmp(C{1},'sensorPosition')
            P.sensorPosition.xOffset = str2double(C{2});  % x offset of the sensor from the center
            P.sensorPosition.yOffset = str2double(C{3});  % y offset of the sensor from the mirror of the car
            P.sensorPosition.zOffset = str2double(C{4});  % Height of the sensor above the floorboard
            P.sensorPosition.yzRot = str2double(C{5}) ; % 0.0 degrees = Rot in y-z plane
            P.sensorPosition.xyRot = str2double(C{6}); % 0.0 degrees = Rot in x-y plane
            P.sensorPosition.xzRot = str2double(C{7}); % 0.0 degrees = Rot in x-z plane
        elseif strcmp(C{1},'fovCfg')
            P.sensorPosition.azimuthFov = str2double(C{3});
            P.sensorPosition.elevationFov = str2double(C{4});
        elseif strcmp(C{1},'numZones')
            P.numZones = str2double(C{2});
        elseif strcmp(C{1},'totNumRows')
            P.totNumRows = str2double(C{2});
        elseif strcmp(C{1},'cuboidDef')
            zoneIdx = str2double(C{2});
            cubeIdx = str2double(C{3});
            if (zoneIdx > P.numZones)
                fprintf('ERROR! numZones %d is less than cuboid zone index %d!\n', P.numZones, zoneIdx);
                exit;
            end
            if ((zoneIdx <= P.numZones) && (cubeIdx <= 3))
                P.zone(zoneIdx).numCuboids        = cubeIdx; %relies on cuboids being defined in rank order
                P.zone(zoneIdx).cuboid(cubeIdx).x = [str2double(C{4}), str2double(C{5})]; %left-right
                P.zone(zoneIdx).cuboid(cubeIdx).y = [str2double(C{6}), str2double(C{7})]; %back-front
                P.zone(zoneIdx).cuboid(cubeIdx).z = [str2double(C{8}), str2double(C{9})]; %floor-ceiling
            end
        elseif strcmp(C{1},'zoneNeighDef')
            zoneIdx = str2double(C{2});
            if (zoneIdx > P.numZones)
                fprintf('ERROR! numZones %d is less than cuboid zone index %d!\n', P.numZones, zoneIdx);
                exit;
            end
            if ((zoneIdx <= P.numZones))
                P.tracker(zoneIdx).zoneType  = str2double(C{3}); 
                numNeigh = str2double(C{4});
                P.tracker(zoneIdx).neighbors = [];
                for id = 1:numNeigh
                   P.tracker(zoneIdx).neighbors(id) = str2double(C{4+id});
                end
            end            
        elseif strcmp(C{1},'occStateMach')
            zoneType = str2double(C{2});
            P.stateMach{zoneType+1}.numPointForEnterThreshold1 = str2double(C{3}); %threshold1 - num points and SNR to enter occupied
            P.stateMach{zoneType+1}.avgSnrForEnterThreshold1   = str2double(C{4});
            P.stateMach{zoneType+1}.numPointForEnterThreshold2 = str2double(C{5}); %threshold2 - num points and SNR to enter occupied
            P.stateMach{zoneType+1}.avgSnrForEnterThreshold2   = str2double(C{6});
            P.stateMach{zoneType+1}.numEntryThreshold = str2double(C{7});
            P.stateMach{zoneType+1}.numPointForStayThreshold   = str2double(C{8}); %number of points and SNR to remain in occupied
            P.stateMach{zoneType+1}.avgSnrForStayThreshold     = str2double(C{9});
            P.stateMach{zoneType+1}.forgetThreshold            = str2double(C{10}); %number of frames with meaningful points to leave occupied
            P.stateMach{zoneType+1}.numPointToForget           = str2double(C{11}); %minimum number of points to not be forget
            P.stateMach{zoneType+1}.overloadThreshold          = str2double(C{12}); %avg SNR to freeze state machine w/large movements
        elseif strcmp(C{1},'interiorBounds')
            P.intBound.minX = str2double(C{2}); %min/max X - width of car interior in meters
            P.intBound.maxX = str2double(C{3});
            P.intBound.minY = str2double(C{4}); %min/max Z - length of car interior in meters
            P.intBound.maxY = str2double(C{5});
        elseif strcmp(C{1},'classParam')
            P.classParam.mode = str2double(C{2});  % mode: 0: disable, 1: enable.
            P.classParam.numFrameAvg = str2double(C{3}); % number of frames that being used for classification
            for zoneId = 1:P.numZones
                P.classParam.thresh_snr(zoneId) = str2double(C{3+zoneId});      % snr threshold
                P.classParam.thresh_vol(zoneId) = str2double(C{3+zoneId+P.numZones});      % volumn threshold
            end
            
        end
        
    end
    P.dataPath.numChirpsPerFrame = (P.frameCfg.chirpEndIdx -...
                                            P.frameCfg.chirpStartIdx + 1) *...
                                            P.frameCfg.numLoops;
    P.dataPath.numDopplerBins = P.dataPath.numChirpsPerFrame / P.dataPath.numTxAnt;
    P.dataPath.numRangeBins = pow2roundup(P.profileCfg.numAdcSamples);
    P.dataPath.rangeResolutionMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
                     (2 * P.profileCfg.freqSlopeConst * 1e12 * P.profileCfg.numAdcSamples);
    P.dataPath.rangeIdxToMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
                     (2 * P.profileCfg.freqSlopeConst * 1e12 * P.dataPath.numRangeBins);
    P.dataPath.dopplerResolutionMps = 3e8 / (2*P.profileCfg.startFreq*1e9 *...
                                        (P.profileCfg.idleTime + P.profileCfg.rampEndTime) *...
                                        1e-6 * P.dataPath.numDopplerBins * P.dataPath.numTxAnt);
end


function [] = dispError()
    disp('Serial Port Error!');
end

function exitPressFcn(hObject, ~)
    setappdata(hObject, 'exitKeyPressed', 1);
end

function [sphandle] = configureDataSport(comPortNum, bufferSize)
    if ~isempty(instrfind('Type','serial'))
        disp('Serial port(s) already open. Re-initializing...');
        delete(instrfind('Type','serial'));  % delete open serial ports.
    end
    comPortString = ['COM' num2str(comPortNum)];
    sphandle = serial(comPortString,'BaudRate',921600);
    set(sphandle,'Terminator', '');
    set(sphandle,'InputBufferSize', bufferSize);
    set(sphandle,'Timeout',15);
    set(sphandle,'ErrorFcn',@dispError);
    fopen(sphandle);
end

function [sphandle] = configureControlPort(comPortNum)
    %if ~isempty(instrfind('Type','serial'))
    %    disp('Serial port(s) already open. Re-initializing...');
    %    delete(instrfind('Type','serial'));  % delete open serial ports.
    %end
    comPortString = ['COM' num2str(comPortNum)];
    sphandle = serial(comPortString,'BaudRate',115200);
    set(sphandle,'Parity','none')    
    set(sphandle,'Terminator','LF')        
    fopen(sphandle);
end


function config = readCfg(filename)
    config = cell(1,100);
    fid = fopen(filename, 'r');
    if fid == -1
        fprintf('File %s not found!\n', filename);
        return;
    else
        fprintf('Opening configuration file %s ...\n', filename);
    end
    tline = fgetl(fid);
    k=1;
    while ischar(tline)
        config{k} = tline;
        tline = fgetl(fid);
        k = k + 1;
    end
    config = config(1:k-1);
    fclose(fid);
end

function length = lengthFromStruct(S)
    fieldName = fieldnames(S);
    length = 0;
    for n = 1:numel(fieldName)
        [~, fieldLength] = S.(fieldName{n});
        length = length + fieldLength;
    end
end

function [R] = readToStruct(S, ByteArray)
    fieldName = fieldnames(S);
    offset = 0;
    for n = 1:numel(fieldName)
        [fieldType, fieldLength] = S.(fieldName{n});
        R.(fieldName{n}) = typecast(uint8(ByteArray(offset+1:offset+fieldLength)), fieldType);
        offset = offset + fieldLength;
    end
end
function CS = validateChecksum(header)
    h = typecast(uint8(header),'uint16');
    a = uint32(sum(h));
    b = uint16(sum(typecast(a,'uint16')));
    CS = uint16(bitcmp(b));
end

function [y] = pow2roundup (x)
    y = 1;
    while x > y
        y = y * 2;
    end
end


function clear_display(figureTab)

    axes('parent', figureTab);
    cla reset;
%    clf;
end


function initialize_occupancy_display(figureTab, Params, xscale)
    global car_bg;
    global person;
    global child;
    global adult; 
    global person2;
    global empty1;
    global empty2;
    global bg_row;
    global bg_col;

    if (Params.totNumRows == 3)
        car_bg = imread('car_3row.png');
    else
        car_bg = imread('car.png');
    end
    empty1 = imread('empty1.png');
    empty2 = imread('empty2.png');
    person = imread('person.png');
    child = imread('child_text.png');
    adult = imread('adult_text.png');
    person2 = imread('person_wht.png');

    [bg_row, bg_col, bg_numClr] = size(car_bg);

    xstart = (1 - xscale) / 2;

    % This creates the 'background' axes
    ax = axes('parent', figureTab, 'units', 'normalized', 'position',[xstart 0 xscale 1]);

    % Move the background axes to the bottom
    uistack(ax,'bottom');
    % Load in a background image and display it using the correct colors
    hi = imagesc(car_bg);
    % Turn the handlevisibility off so that we don't inadvertently plot into the axes again
    % Also, make the axes invisible
    set(ax,'handlevisibility','off', 'visible','off');
    set(ax,'visible', 'off', 'YTick',[],'XTick',[]);
    set(ax,'Color','none');
end

function bg_bottom_selection(source, event)
    global topButtonNewValue;
    global botButtonNewValue;
    global botbt;
    global totNumRows;

    oldValue = botButtonNewValue;

    if totNumRows < 3
        switch event.NewValue.String
            case 'Pause Display',   botButtonNewValue = 1;
            case '2D Point Cloud',  botButtonNewValue = 2;
            case '3D Zones Row 1',  botButtonNewValue = 3;
            case '3D Zones Row 2',  botButtonNewValue = 4;
            case 'Avg Zone SNR',    botButtonNewValue = 5;
            otherwise, botButtonNewValue = 1;
        end
    else
        switch event.NewValue.String
            case 'Pause Display',   botButtonNewValue = 1;
            case '2D Point Cloud',  botButtonNewValue = 2;
            case '3D Zones Row 1',  botButtonNewValue = 3;
            case '3D Zones Row 2',  botButtonNewValue = 4;
            case '3D Zones Row 3',  botButtonNewValue = 5;
            case 'Avg Zone SNR',    botButtonNewValue = 6;
            otherwise, botButtonNewValue = 1;
        end
    end
end


function point3D = transformation(pointCloud, sensorPosition)

        if isempty(pointCloud)
            point3D = [];
            return;
        end
        range = pointCloud(1,:);
        azim  = pointCloud(2,:);
        elev  = pointCloud(3,:);    
        snr   = pointCloud(5,:);

        % Transformation from spherical to
        % cartesian for front mounted sensor
        x = range.*cos(elev).*sin(azim);
        y = range.*cos(elev).*cos(azim);
        z = range.*sin(elev);
        %% rotate back into front mounting position.
        yzRot = sensorPosition.yzRot;
        xyRot = sensorPosition.xyRot;
        xzRot = sensorPosition.xzRot;
        % rotation in yz plane
        y_ = [ cosd(yzRot), sind(yzRot)] * [y; z];
        z_ = [ -sind(yzRot), cosd(yzRot)] * [y; z];
        % rotation in xy plane
        x_ = [ cosd(xyRot), sind(xyRot)] * [x; y_];
        y_ = [ -sind(xyRot), cosd(xyRot)] * [x; y_];
        % rotation in xz plane
        x_ = [ cosd(xzRot), sind(xzRot)] * [x_; z_];
        z_ = [ -sind(xzRot), cosd(xzRot)] * [x_; z_];

        % sensor mounting position adjustment                               
        x_ = x_ + sensorPosition.xOffset;  % offset in left and right
        y_ = y_ + sensorPosition.yOffset;  % offset in front and back
        z_ = z_ + sensorPosition.zOffset;  % Height of the sensor above the floorboard

        point3D = [x_; y_; z_; snr];
end

function [Params, scene] = calcZoneAndSceneBoundary(Params)
scene.minZ = 0; %floor
scene.maxZ = 1.4; % vehicle ceiling
scene.minX = 999.9;
scene.maxX =-999.9;
scene.minY = 999.9;
scene.maxY =-999.9;

for idx = 1:Params.numZones
    zoneMinX =  999.9;
    zoneMaxX = -999.9;
    zoneMinY =  999.9;
    zoneMaxY = -999.9;

    for cIdx = 1:Params.zone(idx).numCuboids
        % First, sanity check the cuboid
        if (Params.zone(idx).cuboid(cIdx).x(1)) > (Params.zone(idx).cuboid(cIdx).x(2))
            temp = Params.zone(idx).cuboid(cIdx).x(1);
            Params.zone(idx).cuboid(cIdx).x(1) = Params.zone(idx).cuboid(cIdx).x(2);
            Params.zone(idx).cuboid(cIdx).x(2) = temp;
        end
        if (Params.zone(idx).cuboid(cIdx).y(1)) > (Params.zone(idx).cuboid(cIdx).y(2))
            temp = Params.zone(idx).cuboid(cIdx).y(1);
            Params.zone(idx).cuboid(cIdx).y(1) = Params.zone(idx).cuboid(cIdx).y(2);
            Params.zone(idx).cuboid(cIdx).y(2) = temp;
        end
        if (Params.zone(idx).cuboid(cIdx).z(1)) > (Params.zone(idx).cuboid(cIdx).z(2))
            temp = Params.zone(idx).cuboid(cIdx).z(1);
            Params.zone(idx).cuboid(cIdx).z(1) = Params.zone(idx).cuboid(cIdx).z(2);
            Params.zone(idx).cuboid(cIdx).z(2) = temp;
        end
        % zg: now it is rotated to the front mount, so z value can be
        % higher than the sensor mounting height. And it is possible to
        % check close to footwell area
        %if (Params.zone(idx).cuboid(cIdx).z(1) < 0.0)  || (Params.zone(idx).cuboid(cIdx).z(2) > sensor.height)
        %    fprintf('Error in zone %d cuboid %d definition\n', idx, cIdx);
        %    return;
        %end
        if (scene.minX > Params.zone(idx).cuboid(cIdx).x(1))
            scene.minX = Params.zone(idx).cuboid(cIdx).x(1);
        end
        if (scene.maxX < Params.zone(idx).cuboid(cIdx).x(2))
            scene.maxX = Params.zone(idx).cuboid(cIdx).x(2);
        end
        if (scene.minY > Params.zone(idx).cuboid(cIdx).y(1))
            scene.minY = Params.zone(idx).cuboid(cIdx).y(1);
        end
        if (scene.maxY < Params.zone(idx).cuboid(cIdx).y(2))
            scene.maxY = Params.zone(idx).cuboid(cIdx).y(2);
        end

        if (zoneMinX > Params.zone(idx).cuboid(cIdx).x(1))
            zoneMinX = Params.zone(idx).cuboid(cIdx).x(1);
        end
        if (zoneMaxX < Params.zone(idx).cuboid(cIdx).x(2))
            zoneMaxX = Params.zone(idx).cuboid(cIdx).x(2);
        end
        if (zoneMinY > Params.zone(idx).cuboid(cIdx).y(1))
            zoneMinY = Params.zone(idx).cuboid(cIdx).y(1);
        end
        if (zoneMaxY < Params.zone(idx).cuboid(cIdx).y(2))
            zoneMaxY = Params.zone(idx).cuboid(cIdx).y(2);
        end
    end

    Params.zone(idx).x_start = zoneMinX;
    Params.zone(idx).x_len = zoneMaxX - zoneMinX;
    Params.zone(idx).y_start = zoneMinY;
    Params.zone(idx).y_len = zoneMaxY - zoneMinY;
end
end