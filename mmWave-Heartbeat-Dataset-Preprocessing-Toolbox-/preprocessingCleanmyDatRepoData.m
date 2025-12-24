clc; clear; 

%% ===================== USER / SYSTEM PARAMETERS =====================
P = getRadarParams();

%% ===================== LOAD RAW ADC DATA =====================
adcData = loadADCData(P);

%% ===================== REORDER INTO CHIRPS =====================
LVDS = reshapeToChirps(adcData, P);

%% ===================== SELECT RX & FRAME CHIRPS =====================
process_adc = selectRXAndChirps(LVDS, P);

%% ===================== RANGE FFT & TARGET BIN SELECTION =====================
[max_bin, deltaR, fft_data] = selectTargetRangeBin(process_adc, P);
% max_bin=max_bin-22;
fprintf('Selected range bin = %d (%.2f m)\n', max_bin, (max_bin-1)*deltaR);

%% ===================== PHASE EXTRACTION =====================
phi = extractVitalPhase(fft_data(:,max_bin));

%% ===================== TIME AXIS =====================
t = (0:length(phi)-1) / P.Fs_vital;

%% ===================== PLOT VITAL-SIGN PHASE =====================
figure;
plot(t, phi, 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Phase (rad)');
title('Radar Vital-Sign Phase (Breathing + Heartbeat)');

%% ===================== EEMD DECOMPOSITION =====================
[breath_wave, heart_wave] = extractBreathHeartEEMD(phi, P.Fs_vital);

%% ===================== PLOT =====================
plotVitalComponents(t, phi, breath_wave, heart_wave);



function P = getRadarParams()

    P.numADCSamples = 200;
    P.numADCBits    = 16;
    P.numTX         = 1;
    P.numRX         = 4;
    P.isReal        = 0;
    
    % Radar
    P.Fs_adc   = 4e6;
    P.c        = 3e8;
    P.slope    = 77.006e12;
    P.startFreq = 60e9;
    P.lambda    = P.c / P.startFreq;
    
    % Chirp / frame timing
    P.framePeriod = 0.05;
    P.Fs_vital    = 1 / P.framePeriod;
    
    % FFT
    P.RangeFFT = 256;
    
    % File
    P.binIdx = 4;
    P.dataSubDir = fullfile('myDatRepoFormat','manavjeet');
end

function adcData = loadADCData(P)
    
    curDir  = fileparts(mfilename('fullpath'));
    dataDir = fullfile(curDir, P.dataSubDir);
    % file    = sprintf('adc_1023gby%d_Raw_0.bin', P.binIdx);
    file    = sprintf('data%d.bin', P.binIdx);
    fname   = fullfile(dataDir, file);
    
    fid = fopen(fname,'r');
    assert(fid~=-1,'Failed to open ADC file');
    
    adcData = fread(fid,'int16');
    fclose(fid);
    
    if P.numADCBits ~= 16
        lmax = 2^(P.numADCBits-1)-1;
        adcData(adcData>lmax) = adcData(adcData>lmax)-2^P.numADCBits;
    end
end

function LVDS = reshapeToChirps(adcData, P)
    
    if P.isReal
        samplesPerChirp = P.numADCSamples * P.numRX;
    else
        samplesPerChirp = 2 * P.numADCSamples * P.numRX;
    end
    
    numChirps = floor(length(adcData)/samplesPerChirp);
    adcData   = adcData(1:numChirps*samplesPerChirp);
    
    if P.isReal
        LVDS = reshape(adcData, P.numADCSamples*P.numRX, numChirps).';
    else
        tmp = zeros(1, length(adcData)/2);
        idx = 1;
        for i = 1:4:length(adcData)-1
            tmp(idx)   = adcData(i)   + 1j*adcData(i+2);
            tmp(idx+1) = adcData(i+1) + 1j*adcData(i+3);
            idx = idx + 2;
        end
        LVDS = reshape(tmp, P.numADCSamples*P.numRX, numChirps).';
    end
end

function process_adc = selectRXAndChirps(LVDS, P)

    numChirps = size(LVDS,1);
    adcAll = zeros(P.numRX, numChirps*P.numADCSamples);
    
    for rx = 1:P.numRX
        for k = 1:numChirps
            adcAll(rx,(k-1)*P.numADCSamples+1:k*P.numADCSamples) = ...
                LVDS(k,(rx-1)*P.numADCSamples+1:rx*P.numADCSamples);
        end
    end
    
    retVal = reshape(adcAll(1,:), P.numADCSamples, numChirps);
    
    numFrames = numChirps / 2;
    process_adc = retVal(:,1:2:end);   % take first chirp per frame
    process_adc = process_adc(:,1:numFrames);
end

function [max_bin, deltaR, fft_data] = selectTargetRangeBin(adc, P)

    fft_data = fft(adc, P.RangeFFT, 1).';
    fft_abs  = abs(fft_data);
    
    deltaR = P.Fs_adc * P.c / (2 * P.slope * P.RangeFFT);
    
    range_energy = zeros(1,P.RangeFFT);
    for r = 1:P.RangeFFT
        d = (r-1)*deltaR;
        if d>0.5 && d<2.5
            range_energy(r) = sum(fft_abs(:,r));
        end
    end
    
    [~, max_bin] = max(range_energy);
end


function phi = extractVitalPhase(binSignal)
    
    phase_raw    = angle(binSignal);
    phase_unwrap = unwrap(phase_raw);
    phase_diff   = [0; diff(phase_unwrap)];
    phi          = smoothdata(phase_diff,'movmean',5);
end

function [breath_wave, heart_wave, breath_idx, heart_idx] = ...
         extractBreathHeartEEMD(phi, Fs_vital)
    
    % phi       : vital-sign phase signal (column or row)
    % Fs_vital  : slow-time sampling rate (Hz)
    
    phi = phi(:);           % enforce column
    N   = length(phi);
    
    %% ---------- EEMD parameters ----------
    NE        = 100;
    noise_amp = 0.2 * std(phi);
    max_imf   = 10;
    
    %% ---------- EEMD ----------
    imf = eemd(phi, noise_amp, NE, max_imf);   % [N × max_imf]
    
    %% ---------- Frequency analysis ----------
    f_axis = (0:N-1) * Fs_vital / N;
    
    breath_idx = [];
    heart_idx  = [];
    
    for k = 1:size(imf,2)
        IMFk = imf(:,k);
    
        F = abs(fft(IMFk));
        Fh = F(1:floor(N/2));
        fh = f_axis(1:floor(N/2));
    
        [~, idx] = max(Fh);
        f_peak = fh(idx);
    
        % Frequency bands
        if f_peak >= 0.1 && f_peak <= 0.5
            breath_idx(end+1) = k;
        elseif f_peak >= 0.8 && f_peak <= 2.0
            heart_idx(end+1) = k;
        end
    end
    
    %% ---------- Reconstruction ----------
    breath_wave = zeros(N,1);
    heart_wave  = zeros(N,1);
    
    if ~isempty(breath_idx)
        breath_wave = sum(imf(:,breath_idx), 2);
    end
    if ~isempty(heart_idx)
        heart_wave  = sum(imf(:,heart_idx), 2);
    end

end

function plotVitalComponents(t, phi, breath_wave, heart_wave)

    figure;
    
    subplot(3,1,1);
    plot(t, phi, 'k'); grid on;
    xlabel('Time (s)');
    ylabel('Phase (rad)');
    title('Original Vital-Sign Phase (Breathing + Heartbeat)');
    
    subplot(3,1,2);
    plot(t, breath_wave, 'b'); grid on;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Breathing Component (EEMD-Reconstructed)');
    
    subplot(3,1,3);
    plot(t, heart_wave, 'r'); grid on;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Heartbeat Component (EEMD-Reconstructed)');

end


