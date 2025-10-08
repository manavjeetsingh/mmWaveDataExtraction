%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%      (C) Copyright 2018-2021 Texas Instruments, Inc.
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
function [] = read_dat_file_and_plot(filename, dim)
%global load_parking_assist load_point_cloud_srr load_point_cloud_usrr load_clusters load_trackers view_range use_perspective_projection;
global load_doppler_range load_y_z load_x_y load_trackers;
global platformType MAX_NUM_OBJECTS OBJ_STRUCT_SIZE_BYTES;
global OBJ_RANSAC_FILTER_BYTES;
global bytevec_log readUartFcnCntr;
global ELEV_VIEW EXIT_KEY_PRESSED BYTE_VEC_ACC_MAX_SIZE bytevecAcc;
global BYTES_AVAILABLE_FCN_CNT BYTES_AVAILABLE_FLAG bytevecAccLen StatsInfo;
global activeFrameCPULoad interFrameCPULoad guiCPULoad guiProcTime;
global loggingEnable fidLog ;
global OBJ_TRACKER_STRUCT_SIZE_BYTES_TARGET view_range_curr;
global OBJ_NOISE_PROFILE_BYTES OBJ_STRUCT_SIZE_BYTES_COMPACT OBJ_NOISE_PROFILE_BYTES_COMPACT;

% The R79 demo works on the awr2944.
platformType = hex2dec('2944');

MMWDEMO_UART_MSG_DETECTED_POINTS    = 1;
MMWDEMO_UART_MSG_NOISE_PROFILE      = 7;
MMWDEMO_UART_MSG_TRACK_OBJ          = 10;
MMWDEMO_UART_MSG_RANSAC_FILTER  = 11;
MMWDEMO_UART_MSG_DETECTED_POINTS_COMPACT  = 12;
MMWDEMO_UART_MSG_NOISE_PROFILE_COMPACT = 13;

load_point_cloud_srr = 1;
load_point_cloud_usrr = 1;
load_clusters  = 0;
load_trackers = 1;
view_range = 0;
load_parking_assist = 0;
view_range_curr = view_range;
load_doppler_range = 1;
load_y_z = 1;
load_x_y = 1;

%% Initialize some constants.
MAX_NUM_OBJECTS = 2000;
OBJ_STRUCT_SIZE_BYTES = 16;
OBJ_TRACKER_STRUCT_SIZE_BYTES_TARGET = 40;
OBJ_NOISE_PROFILE_BYTES = 4;
OBJ_RANSAC_FILTER_BYTES = 1;
OBJ_STRUCT_SIZE_BYTES_COMPACT = 8;
OBJ_NOISE_PROFILE_BYTES_COMPACT = 2;
bytevec_log = zeros(0,1,'single');
readUartFcnCntr = 0;
ELEV_VIEW = 3;
EXIT_KEY_PRESSED = 0;
BYTES_AVAILABLE_FLAG = 0;
BYTES_AVAILABLE_FCN_CNT = 32*8;
BYTE_VEC_ACC_MAX_SIZE  = 2^24;
bytevecAcc = zeros(BYTE_VEC_ACC_MAX_SIZE,1);
bytevecAccLen = 0;

%% Some more Initialisations
StatsInfo.interFrameProcessingTime = 0;
StatsInfo.transmitOutputTime = 0;
StatsInfo.interFrameProcessingMargin = 0;
StatsInfo.interChirpProcessingMargin = 0;
StatsInfo.interFrameCPULoad = 0;
StatsInfo.activeFrameCPULoad = 0;

activeFrameCPULoad = zeros(100,1,'single');
interFrameCPULoad = zeros(100,1,'single');
guiCPULoad = zeros(100,1,'single');
view_range = 0;
guiProcTime = 0;

timeout_ctr         = 0;
%bytevec_cp_max_len  = 2^15;
bytevec_cp_max_len  = 2^24;
bytevec_cp          = zeros(bytevec_cp_max_len,1,'uint8');
bytevec_cp_len      = 0;
loggingEnable       = 0;
fidLog              = 0;
use_perspective_projection = 0;
prev_use_perspective_projection = use_perspective_projection;


%% Initialize the GUI and start.
% filename = '../AWR294X_processed_stream_2021_11_01T23_30_49_678_EgoMobile_RansacTracker_250ms.dat';
% dim.max_dist_x = 70;
% dim.max_dist_y = 250;
% dim.max_dist_z = 20;
% dim.max_vel = 60;
guiMonitor = gui_initializer(dim);
magicNotOkCntr=0;
% Every packet from the AWR2944 has the following barker code at the beginning.
barker_code = char([2 1 4 3 6 5 8 7]);
%-------------------- Main Loop ------------------------
if(1)
    %Read bytes from the UART port. 
    %readUartCallbackFcn(sphandle, 0);
    readFromFile(filename);
	% If bytes are available, append the new bytes to bytevec_cp
    if BYTES_AVAILABLE_FLAG == 1
        BYTES_AVAILABLE_FLAG = 0;
        %fprintf('bytevec_cp_len, bytevecAccLen = %d %d \n',bytevec_cp_len, bytevecAccLen)
        if (bytevec_cp_len + bytevecAccLen) < bytevec_cp_max_len
            bytevec_cp(bytevec_cp_len+1:bytevec_cp_len + bytevecAccLen) = bytevecAcc(1:bytevecAccLen);
            bytevec_cp_len = bytevec_cp_len + bytevecAccLen;
            bytevecAccLen = 0;
        else
            fprintf('Error: Buffer overflow, bytevec_cp_len, bytevecAccLen = %d %d \n',bytevec_cp_len, bytevecAccLen)
            bytevecAccLen = 0;
            bytevec_cp_len = 0;
        end
    end
    
    try 
    bytevecStr = (bytevec_cp);
    magicOk = 0;
	% if the bytevecStr is atleast as large as the header, check if it contains the header. 
    if (bytevec_cp_len > 72) && (size(bytevecStr,2) == 1)
        startIdx = strfind(bytevecStr', barker_code);
    else
        startIdx = [];
    end
    startIdx1 = startIdx;
    len = length(startIdx);
    i=1;
    while(i<=len && ~EXIT_KEY_PRESSED)
        startIdx = startIdx1(i);
        if ~isempty(startIdx)
            if startIdx(1) > 1
                bytevec_cp(1: bytevec_cp_len-(startIdx(1)-1)) = bytevec_cp(startIdx(1):bytevec_cp_len);
                bytevec_cp_len = bytevec_cp_len - (startIdx(1)-1);
                startIdx1 = startIdx1 - startIdx1(i)+1;
            end
            if bytevec_cp_len < 0
                fprintf('Error: %d %d \n',bytevec_cp_len, bytevecAccLen)
                bytevec_cp_len = 0;
            end

            packetlenNum = single(bytevec_cp(8+4+[1:4]));
            totalPacketLen = sum(packetlenNum .* [1 256 65536 16777216]');
            if bytevec_cp_len >= totalPacketLen
                magicOk = 1;
            else
                magicOk = 0;
            end
        end

        byteVecIdx = 0;
        if(magicOk == 1)
            tStart = tic;
            bytevec_cp_flt = single(bytevec_cp);
            % Extract the header. 
            [Header, byteVecIdx] = getHeader(bytevec_cp_flt, byteVecIdx);
            sfIdx = Header.subframeNumber+1;
            if (sfIdx > 2) || (Header.numDetectedObj > MAX_NUM_OBJECTS)
                continue;
            end
            detObj.numObj = 0;
            trackedObj.numTLV = 0;

            % Extract each of the TLVs (type length value) in the current message. 
            for tlvIdx = 1:Header.numTLVs
                [tlv, byteVecIdx] = getTlv(bytevec_cp_flt, byteVecIdx);
                switch tlv.type
                    case MMWDEMO_UART_MSG_DETECTED_POINTS
                        if tlv.length >= OBJ_STRUCT_SIZE_BYTES
                            [detObj, byteVecIdx] = getDetObj(bytevec_cp_flt, Header.numDetectedObj, ...
                                byteVecIdx, ...
                                tlv.length);
                        end
                    case MMWDEMO_UART_MSG_NOISE_PROFILE 
                        if tlv.length >= OBJ_NOISE_PROFILE_BYTES
                            [detObj_noiseProfile, byteVecIdx] = getNoiseProfile(bytevec_cp_flt, Header.numDetectedObj, ...
                                byteVecIdx, ...
                                tlv.length);
                        end
                    case MMWDEMO_UART_MSG_TRACK_OBJ
                        if tlv.length >= OBJ_TRACKER_STRUCT_SIZE_BYTES_TARGET
                            [trackedObj, byteVecIdx] = getTracker(bytevec_cp_flt, ...
                                byteVecIdx, ...
                                tlv.length);
                        end
                    case MMWDEMO_UART_MSG_RANSAC_FILTER 
                        if tlv.length >= OBJ_RANSAC_FILTER_BYTES*Header.numDetectedObj
                            [detObj_RansacFilter, byteVecIdx] = getRansacFilter(bytevec_cp_flt, Header.numDetectedObj, ...
                                byteVecIdx, ...
                                tlv.length);
                        end
                    case MMWDEMO_UART_MSG_DETECTED_POINTS_COMPACT
                        if tlv.length >= OBJ_STRUCT_SIZE_BYTES_COMPACT
                            [detObj, byteVecIdx] = getDetObjCompact(bytevec_cp_flt, Header.numDetectedObj, ...
                                byteVecIdx, ...
                                tlv.length);
                        end
                     case MMWDEMO_UART_MSG_NOISE_PROFILE_COMPACT
                        if tlv.length >= OBJ_NOISE_PROFILE_BYTES_COMPACT
                            [detObj_noiseProfile_comp, byteVecIdx] = getNoiseProfileCompact(bytevec_cp_flt, Header.numDetectedObj, ...
                                byteVecIdx, ...
                                tlv.length);
                        end
                    otherwise
                end
            end

            byteVecIdx = Header.totalPacketLen;
            % Display
            % 1. Detected objects
            if exist('detObj_RansacFilter','var') == 1
                if (detObj.numObj > 0)
                    set(guiMonitor.detectedObjectsPlotHndA, 'Xdata', detObj.x(detObj_RansacFilter==1), 'Ydata', detObj.y(detObj_RansacFilter==1));
                    set(guiMonitor.detectedObjectsPlotHndB, 'Xdata', detObj.x(detObj_RansacFilter==0), 'Ydata', detObj.y(detObj_RansacFilter==0));
                    if load_doppler_range
                        set(guiMonitor.detectedObjectsRngDopPlotHndA, 'Xdata', detObj.range(detObj_RansacFilter==1), 'Ydata', detObj.doppler(detObj_RansacFilter==1));
                        set(guiMonitor.detectedObjectsRngDopPlotHndB, 'Xdata', detObj.range(detObj_RansacFilter==0), 'Ydata', detObj.doppler(detObj_RansacFilter==0));
                    else
                        set(guiMonitor.detectedObjectsRngDopPlotHndA, 'Xdata', inf, 'Ydata', inf);
                        set(guiMonitor.detectedObjectsRngDopPlotHndB, 'Xdata', inf, 'Ydata', inf);
                    end		
                else
                    set(guiMonitor.detectedObjectsPlotHndA, 'Xdata', inf, 'Ydata', inf);
                    set(guiMonitor.detectedObjectsRngDopPlotHndA, 'Xdata', inf, 'Ydata', inf);
                end


                if (detObj.numObj > 0) && load_y_z
                    set(guiMonitor.detectedObjectsPlotYZHndA, 'Xdata', detObj.y(detObj_RansacFilter==1), 'Ydata', detObj.z(detObj_RansacFilter==1));
                    set(guiMonitor.detectedObjectsPlotYZHndB, 'Xdata', detObj.y(detObj_RansacFilter==0), 'Ydata', detObj.z(detObj_RansacFilter==0));
                else
                    set(guiMonitor.detectedObjectsPlotYZHndA, 'Xdata', inf, 'Ydata', inf);
                    set(guiMonitor.detectedObjectsPlotYZHndB, 'Xdata', inf, 'Ydata', inf);
                end
            else
                if (detObj.numObj > 0)
                    set(guiMonitor.detectedObjectsPlotHndA, 'Xdata', detObj.x, 'Ydata', detObj.y);
                    %set(guiMonitor.detectedObjectsPlotHndB, 'Xdata', detObj.x, 'Ydata', detObj.y);
                      if load_doppler_range
                        set(guiMonitor.detectedObjectsRngDopPlotHndA, 'Xdata', detObj.range, 'Ydata', detObj.doppler);
                        %set(guiMonitor.detectedObjectsRngDopPlotHndB, 'Xdata', detObj.range, 'Ydata', detObj.doppler);
                      else
                        set(guiMonitor.detectedObjectsRngDopPlotHndA, 'Xdata', inf, 'Ydata', inf);
                        %set(guiMonitor.detectedObjectsRngDopPlotHndB, 'Xdata', inf, 'Ydata', inf);
                      end		
                else
                    %set(guiMonitor.detectedObjectsPlotHndA, 'Xdata', inf, 'Ydata', inf);
                    set(guiMonitor.detectedObjectsRngDopPlotHndA, 'Xdata', inf, 'Ydata', inf);
                end
                if (detObj.numObj > 0) && load_y_z
                    set(guiMonitor.detectedObjectsPlotYZHndA, 'Xdata', detObj.y, 'Ydata', detObj.z);
                    %set(guiMonitor.detectedObjectsPlotYZHndB, 'Xdata', detObj.y, 'Ydata', detObj.z);
                    else
                    set(guiMonitor.detectedObjectsPlotYZHndA, 'Xdata', inf, 'Ydata', inf);
                    %set(guiMonitor.detectedObjectsPlotYZHndB, 'Xdata', inf, 'Ydata', inf);
                end
            end

            % 2. Tracking
            if exist('trackedObj','var')
                if (trackedObj.numTLV > 0)
                    if load_x_y
                        set(guiMonitor.trackedObjPlotHnd, 'Xdata', trackedObj.posX, 'Ydata', trackedObj.posY);
                    end
                    if load_doppler_range
                        set(guiMonitor.trackedObjRngDop, 'Xdata', trackedObj.range, 'Ydata', trackedObj.doppler);
                    else
                        set(guiMonitor.trackedObjRngDop, 'Xdata', inf, 'Ydata', inf);
                    end
                end
            end
            guiProcTime = round(toc(tStart) * 1e3);
        else
            magicNotOkCntr = magicNotOkCntr + 1;
        end

        if bytevec_cp_len > (bytevec_cp_max_len * 7/8)
            bytevec_cp_len = 0;
        end

%         % Update near view. 
%         if (view_range_curr ~= view_range)
%             if view_range == 0
%                 range_depth_tmp = dim.max_dist_y;`
%                 range_width_tmp = dim.max_dist_x;
%                 dopplerRange_tmp = dim.max_vel;
%             else
%                 range_depth_tmp = dim.max_dist_y/4;
%                 range_width_tmp = dim.max_dist_x/4;
%                 dopplerRange_tmp = dim.max_vel/4;
%             end
%             view_range_curr = view_range;
% 
%             subplot(guiMonitor.detectedObjectsFigHnd);
%             axis([-range_width_tmp range_width_tmp 0 range_depth_tmp]);
%             subplot(guiMonitor.detectedObjectsRngDopFigHnd);
%             axis([0 range_depth_tmp -dopplerRange_tmp dopplerRange_tmp]);
%             subplot(guiMonitor.detectedObjectsYZFigHnd);
%             axis([0 range_depth_tmp -dim.max_dist_z/4 dim.max_dist_z ]) 
% 
%         end
% 
%         if use_perspective_projection ~= prev_use_perspective_projection
%             camproj(guiMonitor.detectedObjectsFigHnd, 'perspective');
%             campos(guiMonitor.detectedObjectsFigHnd, [0,0,dim.max_dist_y/2]);
%             camtarget([0,dim.max_dist_y*0.33,0]);
%             prev_use_perspective_projection = use_perspective_projection;
%         end
        tIdleStart = tic;

%         if(toc(tIdleStart) > 2*Params(1).frameCfg.framePeriodicity/1000)
%             timeout_ctr=timeout_ctr+1;
%             tIdleStart = tic;
%         end

        i=i+1;
        pause(0.3);
    
    end
    
    catch
        disp('Recovering...');
        
        if byteVecIdx > 0
            shiftSize = byteVecIdx;
            bytevec_cp = bytevec_cp(shiftSize+1:bytevec_cp_len);
            bytevec_cp_len = bytevec_cp_len - shiftSize;
            if bytevec_cp_len < 0
                fprintf('Error: bytevec_cp_len < bytevecAccLen, %d %d \n', bytevec_cp_len, bytevecAccLen)
                bytevec_cp_len = 0;
            end
        end
    end    
    pause(0.3);    
end
close(guiMonitor.figHnd); % close figure
delete(guiMonitor.figHnd);
%return

function [] = readFromFile(filename)
global bytevecAcc;
global bytevecAccLen;
global readUartFcnCntr;
global BYTES_AVAILABLE_FLAG;
global BYTE_VEC_ACC_MAX_SIZE;

% [bytevec, byteCount] = fread(obj, bytesToRead, 'uint8');
%fid = fopen('../AWR294X_processed_stream_2021_10_19T09_06_05_807.dat','r');
%fid = fopen('../AWR294X_processed_stream_2021_10_22T22_50_20_336.dat','r');
fid = fopen(filename,'r');
[bytevec, byteCount] = fread(fid,inf,'*uint8');

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


function [y] = pow2roundup (x)
y = 1;
while x > y
    y = y * 2;
end
return


function [Header, idx] = getHeader(bytevec, idx)
idx = idx + 8; %Skip magic word
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
Header.subframeNumber = sum(bytevec(idx+[1:4]) .* word);
idx = idx + 4;
return

function [tlv, idx] = getTlv(bytevec, idx)
word = [1 256 65536 16777216]';
tlv.type = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
tlv.length = sum(bytevec(idx+(1:4)) .* word);
idx = idx + 4;
return


function [detObj, idx] = getDetObj(bytevec, numDetectedObj, idx, tlvLen)
global OBJ_STRUCT_SIZE_BYTES;
detObj.x = [];
detObj.y = [];
detObj.z = [];
detObj.doppler = [];
detObj.numObj = 0;
len_bytevec = length(bytevec);
if len_bytevec < idx + 4 
    idx = len_bytevec;
    return;
end
if tlvLen > 0
    detObj.numObj = numDetectedObj;
    %Check if the array can be fulfilled. 
    if len_bytevec < idx + detObj.numObj*OBJ_STRUCT_SIZE_BYTES
        detObj.numObj = 0; 
        idx = len_bytevec;
        return;
    end
    bytes = bytevec(idx+(1:detObj.numObj*OBJ_STRUCT_SIZE_BYTES));
    idx = idx + detObj.numObj*OBJ_STRUCT_SIZE_BYTES;
    
    bytes = reshape(bytes, OBJ_STRUCT_SIZE_BYTES, detObj.numObj);
    
    for i=1:size(bytes,2)
        detObj.x = [detObj.x (typecast(uint8([bytes(1:4,i)]),'single'))];
        detObj.y = [detObj.y (typecast(uint8([bytes(5:8,i)]),'single'))];
        detObj.z = [detObj.z (typecast(uint8([bytes(9:12,i)]),'single'))];
        detObj.doppler = [detObj.doppler (typecast(uint8([bytes(13:16,i)]),'single'))];
    end

    detObj.range = sqrt(detObj.y.*detObj.y + detObj.x.*detObj.x);
end
return

function [detObj_noiseProfile, idx] = getNoiseProfile(bytevec, numDetectedObj, idx, tlvLen)
global OBJ_NOISE_PROFILE_BYTES;
detObj_noiseProfile.snr = [];
detObj_noiseProfile.noise = [];
detObj_noiseProfile.numObj = 0;
len_bytevec = length(bytevec);
if len_bytevec < idx + 2 
    idx = len_bytevec;
    return;
end
if tlvLen > 0
    detObj_noiseProfile.numObj = numDetectedObj;
    %Check if the array can be fulfilled. 
    if len_bytevec < idx + detObj_noiseProfile.numObj*OBJ_NOISE_PROFILE_BYTES
        detObj_noiseProfile.numObj = 0; 
        idx = len_bytevec;
        return;
    end
    bytes = bytevec(idx+(1:detObj_noiseProfile.numObj*OBJ_NOISE_PROFILE_BYTES));
    idx = idx + detObj_noiseProfile.numObj*OBJ_NOISE_PROFILE_BYTES;
    
    bytes = reshape(bytes, OBJ_NOISE_PROFILE_BYTES, detObj_noiseProfile.numObj);
    
    for i=1:size(bytes,2)
        detObj_noiseProfile.snr = [detObj_noiseProfile.snr (typecast(uint8([bytes(1:2,i)]),'int16'))];
        detObj_noiseProfile.noise = [detObj_noiseProfile.noise (typecast(uint8([bytes(3:4,i)]),'int16'))];
    end

end
return

function [trackObj, idx] = getTracker(bytevec, idx, tlvLen)
global OBJ_TRACKER_STRUCT_SIZE_BYTES_TARGET;
trackObj.posX = [];
trackObj.posY = [];
trackObj.posZ = [];
trackObj.velX = [];
trackObj.velY = [];
trackObj.velZ = [];
trackObj.accX = [];
trackObj.accY = [];
trackObj.accZ = [];
trackObj.state = [];
len_bytevec = length(bytevec);
if len_bytevec < idx + 4 
    idx = len_bytevec;
    return;
end
if tlvLen > 0
    trackObj.numTLV = tlvLen/OBJ_TRACKER_STRUCT_SIZE_BYTES_TARGET;
    %Check if the array can be fulfilled. 
    if len_bytevec < idx + trackObj.numTLV*OBJ_TRACKER_STRUCT_SIZE_BYTES_TARGET
        trackObj.numTLV = 0; 
        idx = len_bytevec;
        return;
    end
    bytes = bytevec(idx+(1:trackObj.numTLV*OBJ_TRACKER_STRUCT_SIZE_BYTES_TARGET));
    idx = idx + trackObj.numTLV*OBJ_TRACKER_STRUCT_SIZE_BYTES_TARGET;
    
    bytes = reshape(bytes, OBJ_TRACKER_STRUCT_SIZE_BYTES_TARGET, trackObj.numTLV);
    for i=1:size(bytes,2)
        trackObj.posX = [trackObj.posX (typecast(uint8([bytes(1:4,i)]),'single'))];
        trackObj.posY = [trackObj.posY (typecast(uint8([bytes(5:8,i)]),'single'))];
        trackObj.posZ = [trackObj.posZ (typecast(uint8([bytes(9:12,i)]),'single'))];
        trackObj.velX = [trackObj.velX (typecast(uint8([bytes(13:16,i)]),'single'))];
        trackObj.velY = [trackObj.velY (typecast(uint8([bytes(17:20,i)]),'single'))];
        trackObj.velZ = [trackObj.velZ (typecast(uint8([bytes(21:24,i)]),'single'))];
        trackObj.accX = [trackObj.accX (typecast(uint8([bytes(25:28,i)]),'single'))];
        trackObj.accY = [trackObj.accY (typecast(uint8([bytes(29:32,i)]),'single'))];
        trackObj.accZ = [trackObj.accZ (typecast(uint8([bytes(33:36,i)]),'single'))];
        trackObj.state = [trackObj.state (typecast(uint8([bytes(37:40,i)]),'uint32'))];
        trackObj.range = sqrt(trackObj.posY.*trackObj.posY + trackObj.posX.*trackObj.posX);
        trackObj.doppler = (trackObj.velY.*trackObj.posY + trackObj.velX.*trackObj.posX)./trackObj.range;
    end
end
return

function [detObj_RansacFilter, idx] = getRansacFilter(bytevec, numDetectedObj, idx, tlvLen)
global OBJ_RANSAC_FILTER_BYTES;
detObj_RansacFilter = [];
len_bytevec = length(bytevec);
if len_bytevec < idx + 2 
    idx = len_bytevec;
    return;
end
if tlvLen > 0
    detObj_RansacFilter_numObj = numDetectedObj;
    %Check if the array can be fulfilled. 
    if len_bytevec < idx + detObj_RansacFilter_numObj*OBJ_RANSAC_FILTER_BYTES;
        detObj_RansacFilter_numObj = 0; 
        idx = len_bytevec;
        return;
    end
    bytes = bytevec(idx+(1:detObj_RansacFilter_numObj*OBJ_RANSAC_FILTER_BYTES));
    idx = idx + detObj_RansacFilter_numObj*OBJ_RANSAC_FILTER_BYTES;
    
    bytes = reshape(bytes, OBJ_RANSAC_FILTER_BYTES, detObj_RansacFilter_numObj);
    
    for i=1:size(bytes,2)
        detObj_RansacFilter = [detObj_RansacFilter (typecast(uint8([bytes(1,i)]),'uint8'))];
    end
end
return

function [detObj, idx] = getDetObjCompact(bytevec, numDetectedObj, idx, tlvLen)
global OBJ_STRUCT_SIZE_BYTES_COMPACT;
detObj.azimSPQ = [];
detObj.elevSPQ = [];
detObj.rangeIdx = [];
detObj.dopplerIdx = [];
detObj.numObj = 0;
len_bytevec = length(bytevec);
if len_bytevec < idx + 4 
    idx = len_bytevec;
    return;
end
if tlvLen > 0
    detObj.numObj = numDetectedObj;
    %Check if the array can be fulfilled. 
    if len_bytevec < idx + detObj.numObj*OBJ_STRUCT_SIZE_BYTES_COMPACT
        detObj.numObj = 0; 
        idx = len_bytevec;
        return;
    end
    bytes = bytevec(idx+(1:detObj.numObj*OBJ_STRUCT_SIZE_BYTES_COMPACT));
    idx = idx + detObj.numObj*OBJ_STRUCT_SIZE_BYTES_COMPACT;
    
    bytes = reshape(bytes, OBJ_STRUCT_SIZE_BYTES_COMPACT, detObj.numObj);
    
    for i=1:size(bytes,2)
        detObj.azimSPQ = [detObj.azimSPQ (typecast(uint8([bytes(1:2,i)]),'int16'))];
        detObj.elevSPQ = [detObj.elevSPQ (typecast(uint8([bytes(3:4,i)]),'int16'))];
        detObj.rangeIdx = [detObj.rangeIdx (typecast(uint8([bytes(5:6,i)]),'int16'))];
        detObj.dopplerIdx = [detObj.dopplerIdx (typecast(uint8([bytes(7:8,i)]),'int16'))];
    end
    
    rangeStep = 1.32637119;
    dopplerStep = 0.106392793;
    
    azimSinPhase = double(detObj.azimSPQ)/32767.0; 
    azimCosPhase = sqrt(1 - azimSinPhase.^2);
    elevSinPhase = double(detObj.elevSPQ)/32767.0; 
    elevCosPhase = sqrt(1 - elevSinPhase.^2);
    
    detObj.range = double(detObj.rangeIdx) * rangeStep;
    detObj.z = detObj.range .* elevSinPhase;
    detObj.x = detObj.range .* elevCosPhase .* azimSinPhase;
    detObj.y = detObj.range .* elevCosPhase .* azimCosPhase;
    detObj.doppler = double(detObj.dopplerIdx) .* dopplerStep;
end
return

function [detObj_noiseProfile, idx] = getNoiseProfileCompact(bytevec, numDetectedObj, idx, tlvLen)
global OBJ_NOISE_PROFILE_BYTES_COMPACT;
detObj_noiseProfile.snr = [];
detObj_noiseProfile.numObj = 0;
len_bytevec = length(bytevec);
if len_bytevec < idx + 2 
    idx = len_bytevec;
    return;
end
if tlvLen > 0
    detObj_noiseProfile.numObj = numDetectedObj;
    %Check if the array can be fulfilled. 
    if len_bytevec < idx + detObj_noiseProfile.numObj*OBJ_NOISE_PROFILE_BYTES_COMPACT
        detObj_noiseProfile.numObj = 0; 
        idx = len_bytevec;
        return;
    end
    bytes = bytevec(idx+(1:detObj_noiseProfile.numObj*OBJ_NOISE_PROFILE_BYTES_COMPACT));
    idx = idx + detObj_noiseProfile.numObj*OBJ_NOISE_PROFILE_BYTES_COMPACT;
    
    bytes = reshape(bytes, OBJ_NOISE_PROFILE_BYTES_COMPACT, detObj_noiseProfile.numObj);
    
    for i=1:size(bytes,2)
        detObj_noiseProfile.snr = [detObj_noiseProfile.snr (typecast(uint8([bytes(1:2,i)]),'int16'))];
    end

end
return
