clear;
sampling_frequency = 10000 * 1e3; % Hz 
sweepSlope=29.982 * 1e12; % mhz/us
center_freqp=77*1e9;
ramp_end=60; % us
idle_time=100; % us
chirp_cycle_time=(idle_time + ramp_end) * 1e-6;
pulse_repetation_interval = 1/chirp_cycle_time;
num_rx=1;
num_chirps=1;
samples_per_chirp=256;

fileName="Sitting1Tx1Rx/static_sitting_Raw_0.bin";
data=readDCA1000(fileName);

% % Create phased.RangeResponse System object that performs range filtering
% % on fast-time (range) data, using an FFT-based algorithm
% rangeresp = phased.RangeResponse(RangeMethod = 'FFT', ...
%     RangeFFTLengthSource = 'Property', ...
%     RangeFFTLength = samples_per_chirp, ...
%     SampleRate = sampling_frequency, ...
%     SweepSlope = sweepSlope, ...
%     ReferenceRangeCentered = false);
% 
% % % Create range doppler scope to compute and display the response map.
% % rdscope = phased.RangeDopplerScope(IQDataInput=true, ...
% %     SweepSlope = sweepSlope,SampleRate = fs, ...
% %     DopplerOutput="Speed",OperatingFrequency=fc, ...
% %     PRFSource="Property",PRF=prf, ...
% %     RangeMethod="FFT",RangeFFTLength=nr, ...
% %     ReferenceRangeCentered = false);
% 
% plotResponse(rangeresp, data');


% Define parameters for the signal
Fs = sampling_frequency;         % Sampling frequency (Hz)
T = 1/Fs;         % Sampling period (s)
L = samples_per_chirp;         % Length of signal
t = (0:L-1)*T;    % Time vector

Y=fft(data(:, 1:256))
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1); 
f = Fs*(0:(L/2))/L;
distances=f*3e8/2/sweepSlope
figure;
plot(distances,P1);
title('Single-Sided Amplitude Spectrum of x(t)');
% xlabel('Frequency (Hz)');
xlabel('Distance (m)');
ylabel('|P1(f)|');
grid on;