% Helper functions to read, parse and send configuration to device 
% Functions are moved from Occupancy_detection_3D_visualizer.m 


% read configurations from the file 
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

% Parse the configuration file 
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
        % KNS - replaces the profile config for 60Low devices 
        elseif strcmp(C{1},'chirpComnCfg')
            P.chirpComnCfg.DigOutputSampRate_Dcim = str2double(C{2});
            P.chirpComnCfg.DigOutputBitsSel = str2double(C{3});
            P.profileCfg.digOutSampleRate = P.chirpComnCfg.DigOutputBitsSel; 
            P.chirpComnCfg.DfeFirSel = str2double(C{4});
            P.chirpComnCfg.NumOfAdcSamples = str2double(C{5});
            P.profileCfg.numAdcSamples = P.chirpComnCfg.NumOfAdcSamples;
            P.chirpComnCfg.ChirpTxMimoPatSel = str2double(C{6});
            P.chirpComnCfg.MiscSettings = str2double(C{7});
            P.chirpComnCfg.HpfFastInitDuration = str2double(C{8});
            P.chirpComnCfg.CrdNSlopeMag = str2double(C{9});
            P.chirpComnCfg.ChirpRampEndTime = str2double(C{10});
            P.profileCfg.rampEndTime = P.chirpComnCfg.ChirpRampEndTime; 
            P.chirpComnCfg.ChirpRxHpfSel = str2double(C{11});
        elseif strcmp(C{1},'chirpTimingCfg')
            P.chirpTimingCfg.ChirpIdleTime = str2double(C{2});
            P.profileCfg.idleTime = P.chirpTimingCfg.ChirpIdleTime; 
            P.chirpTimingCfg.ChirpAdcSkipSamples = str2double(C{3});
            P.chirpTimingCfg.ChirpTxStartTime = str2double(C{4});
            P.chirpTimingCfg.ChirpRfFreqSlope = str2double(C{5});
            P.profileCfg.freqSlopeConst = P.chirpTimingCfg.ChirpRfFreqSlope; 
            P.chirpTimingCfg.ChirpRfFreqStart = str2double(C{6});
            P.profileCfg.startFreq = P.chirpTimingCfg.ChirpRfFreqStart;
            P.chirpTimingCfg.ChirpTxEnSel = str2double(C{7});
            P.chirpTimingCfg.ChirpTxBpmEnSel = str2double(C{8});
        
            
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
        end
    end
    % KNS - since these parametes are for diaplay only, just ignore for now
    % 
%     P.dataPath.numChirpsPerFrame = (P.frameCfg.chirpEndIdx -...
%                                             P.frameCfg.chirpStartIdx + 1) *...
%                                             P.frameCfg.numLoops;
%     P.dataPath.numDopplerBins = P.dataPath.numChirpsPerFrame / P.dataPath.numTxAnt;
%     P.dataPath.numRangeBins = pow2roundup(P.profileCfg.numAdcSamples);
%     P.dataPath.rangeResolutionMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
%                      (2 * P.profileCfg.freqSlopeConst * 1e12 * P.profileCfg.numAdcSamples);
%     P.dataPath.rangeIdxToMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
%                      (2 * P.profileCfg.freqSlopeConst * 1e12 * P.dataPath.numRangeBins);
%     P.dataPath.dopplerResolutionMps = 3e8 / (2*P.profileCfg.startFreq*1e9 *...
%                                         (P.profileCfg.idleTime + P.profileCfg.rampEndTime) *...
%                                         1e-6 * P.dataPath.numDopplerBins * P.dataPath.numTxAnt);

    P.dataPath.numChirpsPerFrame = 0;
    P.dataPath.numDopplerBins = 0;
    P.dataPath.numRangeBins = 0;
    P.dataPath.rangeResolutionMeters = 0;
    P.dataPath.rangeIdxToMeters = 0;
    P.dataPath.dopplerResolutionMps = 0;
end
