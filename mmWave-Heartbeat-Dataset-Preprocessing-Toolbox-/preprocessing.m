%% Single-person breathing + heartbeat phase vital-sign extraction (output only one time-domain plot)
% Hardware: IWR6843ISK + DCA1000
% Data: 1024 frames, 2 chirps per frame, take the 1st chirp of each frame
% Frame period = 50 ms => slow-time sampling rate = 20 Hz

clc; clear; 

%% ================== Basic parameters ==================
numADCSamples = 200;     % Number of ADC samples per chirp
numADCBits    = 16;      % ADC resolution (bits)
numTX         = 1;       % Number of TX antennas
numRX         = 4;       % Number of RX antennas
isReal        = 0;       % 0 = complex sampling (I/Q), 1 = real sampling

% Radar parameters
Fs_adc   = 4e6;          % ADC sampling rate (Hz)
c        = 3e8;          % Speed of light (m/s)
ts       = numADCSamples / Fs_adc;   % Sampling time of one chirp
slope    = 77.006e12;    % FMCW slope (Hz/s)
B_valid  = ts * slope;   % Effective bandwidth
deltaR0  = c / (2*B_valid);     % Original range resolution (for 200 samples), ~4.6 cm
startFreq = 60e9;        % Start frequency (Hz)
lambda    = c / startFreq;

% Slow-time sampling rate (vital-sign signal)
framePeriod = 0.05;      % 50 ms per frame (given configuration)
Fs_vital    = 1/framePeriod;   % 20 Hz

%% ================== Read bin file (relative path) ==================
% Directory of the current script, e.g., ...\dataset_exp2024\1023data
curDir = fileparts(mfilename('fullpath'));

% Subdirectory containing bin files: 1023dataset\gby1023data
dataDir = fullfile(curDir, '1023dataset', 'gby1023data');

% Select which gby data to read: adc_1023gby1_Raw_0.bin
binIdx = 1;
fileNameOnly = sprintf('adc_1023gby%d_Raw_0.bin', binIdx);
Filename = fullfile(dataDir, fileNameOnly);

[fid, msg] = fopen(Filename, 'r');
if fid == -1
    error('Unable to open data file: %s\nSystem message: %s', Filename, msg);
end
adcDataRow = fread(fid, 'int16');
fclose(fid);

% If ADC resolution is not 16 bits, apply sign extension (usually unnecessary here)
if numADCBits ~= 16
    l_max = 2^(numADCBits-1)-1;
    adcDataRow(adcDataRow > l_max) = adcDataRow(adcDataRow > l_max) - 2^numADCBits;
end

%% ================== Reorder data by chirp ==================
fileSize = length(adcDataRow);     % Total length in int16

if isReal
    % Real sampling: samples per chirp
    samplesPerChirp = numADCSamples * numRX;
else
    % Complex sampling: I/Q channels
    samplesPerChirp = 2 * numADCSamples * numRX;
end

numChirps_total = floor(fileSize / samplesPerChirp);
fileSize        = numChirps_total * samplesPerChirp;    % Truncate to integer chirps
adcData         = adcDataRow(1:fileSize);

if isReal
    numChirps = fileSize / (numADCSamples * numRX);
    LVDS = reshape(adcData, numADCSamples*numRX, numChirps).';   % [numChirps, numADCSamples*numRX]
else
    % Complex sampling: combine I/Q
    numChirps = fileSize / (2 * numADCSamples * numRX);
    LVDS = zeros(1, fileSize/2);
    counter = 1;
    for i = 1:4:fileSize-1   % Data format: I0,Q0,I1,Q1,...
        LVDS(counter)   = adcData(i)   + 1j*adcData(i+2);
        LVDS(counter+1) = adcData(i+1) + 1j*adcData(i+3);
        counter = counter + 2;
    end
    LVDS = reshape(LVDS, numADCSamples*numRX, numChirps).';      % [numChirps, numADCSamples*numRX]
end

%% ================== Select RX channel 1 and keep only the first chirp of each frame ==================
% At this point: numChirps = 2048 (1024 frames × 2 chirps)
adcAll = zeros(numRX, numChirps*numADCSamples);
for rx = 1:numRX
    for k = 1:numChirps
        adcAll(rx, (k-1)*numADCSamples+1 : k*numADCSamples) = ...
            LVDS(k, (rx-1)*numADCSamples+1 : rx*numADCSamples);
    end
end

% Use only the first RX antenna
retVal = reshape(adcAll(1,:), numADCSamples, numChirps);   % [200, 2048]

% Two chirps per frame, keep only the first one => 1024 chirps
numFrames   = numChirps/2;
process_adc = zeros(numADCSamples, numFrames);
for n = 1:2:numChirps
    process_adc(:, (n+1)/2) = retVal(:, n);
end
% process_adc: [200, 1024], one chirp per column (slow-time rate = 20 Hz)

%% ================== Range FFT and select the human range bin ==================
RangFFT = 256;
adcdata = process_adc;            % [200, 1024]

fft_data = fft(adcdata, RangFFT, 1);  % FFT along range dimension
fft_data = fft_data.';                % [1024, 256]: rows = chirps, columns = range bins
fft_abs  = abs(fft_data);

% Range resolution after zero-padding to 256 points
deltaR = Fs_adc * c / (2 * slope * RangFFT);

% Non-coherent accumulation in 0.5–2.5 m range to find max-energy bin
range_energy = zeros(1, RangFFT);
range_max    = 0;
max_bin      = 1;

for r = 1:RangFFT
    dist_r = (r-1) * deltaR;
    if dist_r > 0.5 && dist_r < 2.5
        range_energy(r) = sum(fft_abs(:, r));
        if range_energy(r) > range_max
            range_max = range_energy(r);
            max_bin   = r;
        end
    end
end

fprintf('Selected human range bin = %d, approx. %.2f m\n', ...
        max_bin, (max_bin-1)*deltaR);

%% ================== Extract phase of the selected bin, unwrap + diff + smoothing ==================
% Raw phase in [-pi, pi]
phase_raw = angle(fft_data(:, max_bin));    % 1024×1

% Phase unwrapping for continuity
phase_unwrap = unwrap(phase_raw);

% Phase differentiation: suppress slow drift, enhance heartbeat while preserving breathing
phase_diff = [0; diff(phase_unwrap)];       % Pad first sample with 0

% Moving average smoothing (window = 5) to remove impulse noise
phi = smoothdata(phase_diff, 'movmean', 5); % Vital-sign phase (breathing + heartbeat)

%% ================== Build real time axis and plot ==================
t = (0:length(phi)-1) / Fs_vital;   % Real time (seconds)

figure;
plot(t, phi, 'LineWidth', 1.5);
grid on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Phase (rad)', 'FontSize', 12);
title(sprintf('Radar Vital-Sign Phase (Breathing + Heartbeat), gby%d', binIdx), ...
      'FontSize', 14);

%% ================== EEMD decomposition: breathing / heartbeat components ==================
% Use the overall vital-sign phase signal phi (after diff + smoothing)

Fs_vital = 1/0.05;          % Slow-time sampling rate = 20 Hz (same as above)
phi_sig  = phi(:)';         % Convert to row vector (required by some EEMD functions)

% ---- EEMD parameters (can be tuned) ----
NE        = 100;               % Ensemble number
noise_amp = 0.2*std(phi_sig); % Noise amplitude (typically 0.1–0.3 × std)
max_imf   = 10;                % Maximum number of IMFs

% Call EEMD (adjust parameter order if required by your toolbox)
% Common usage: imf = eemd(x, noise_amp, NE, max_imf);
imf = eemd(phi_sig, noise_amp, NE, max_imf);

% If EEMD returns IMF × N, transpose to N × IMF
if size(imf,1) < size(imf,2)
    imf = imf.';
end

[N,K] = size(imf);

% ---- Identify breathing / heartbeat IMFs based on dominant frequency ----
breath_idx = [];   % Breathing IMF indices
heart_idx  = [];   % Heartbeat IMF indices

f_axis = (0:N-1)*(Fs_vital/N);  % Frequency axis

for k = 1:K
    IMFk = imf(:,k);
    IMF_fft = abs(fft(IMFk));
    IMF_fft_half = IMF_fft(1:floor(N/2));
    f_half       = f_axis(1:floor(N/2));

    [~, idx_max] = max(IMF_fft_half);
    f_peak = f_half(idx_max);   % Dominant frequency of this IMF

    % Frequency-based classification:
    % Breathing: ~0.1–0.5 Hz
    % Heartbeat: ~0.8–2 Hz
    if f_peak >= 0.1 && f_peak <= 0.5
        breath_idx = [breath_idx, k];
    elseif f_peak >= 0.8 && f_peak <= 2
        heart_idx = [heart_idx, k];
    end
end

disp('EEMD-detected breathing-related IMFs:');
disp(breath_idx);
disp('EEMD-detected heartbeat-related IMFs:');
disp(heart_idx);

% ---- Reconstruct breathing / heartbeat waveforms ----
breath_wave = zeros(N,1);
heart_wave  = zeros(N,1);

if ~isempty(breath_idx)
    breath_wave = sum(imf(:, breath_idx), 2);
end
if ~isempty(heart_idx)
    heart_wave = sum(imf(:, heart_idx), 2);
end

% ---- Plot: original vital-sign vs breathing vs heartbeat ----
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
