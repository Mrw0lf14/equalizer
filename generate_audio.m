clear, clc, close all;
fs = 44100;
ts = 0: 1/fs: 0.25-1/fs;

audio = sin(2*pi*500*ts);
for i = 2:40
    audio = [audio sin(2*pi*500*i*ts)];
end
N = length(audio);
f = 0: fs/N : fs-fs/N;
fft_audio = 2*abs(fft(audio))/N;
figure;
plot(f, fft_audio), grid on;
audiowrite("in.wav", audio, fs);
