import numpy as np

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        N = N - 1
        # raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])
    
# Задаем входной сигнал (в данном примере синусоида с частотой 10 Гц)
fs = 1000  # Частота дискретизации
t = np.linspace(0, 1, fs)  # Временной промежуток 1 секунда
signal = np.sin(2 * np.pi * 200 * t) + 1/2*np.sin(2 * np.pi * 100 * t)  # 10 Гц синусоида
N = len(signal)
# Преобразование Фурье
fft_signal = np.fft.fft(signal)
fft_signal2 = FFT(signal)

print(len(fft_signal))
freqs = np.linspace(0, fs-1/fs, N)
print(len(freqs))
# Обратное преобразование Фурье
reconstructed_signal = np.fft.ifft(fft_signal)
#print(np.abs(fft_signal))
# Оставляем значения от нуля до частоты дискретизации
# fft_signal[int(fs/2):] = 0

# Обратное преобразование Фурье для нового сигнала
reconstructed_signal_filtered = np.fft.ifft(fft_signal)

# Визуализация результатов
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, signal)
plt.title('Входной сигнал')

plt.subplot(2, 2, 2)
plt.plot(np.abs(fft_signal))
plt.title('Спектр сигнала (от нуля до fs)')

plt.subplot(2, 2, 3)
plt.plot(np.abs(fft_signal2))
plt.title('Восстановленный сигнал (полное преобразование)')

plt.subplot(2, 2, 4)
plt.plot(t, np.real(reconstructed_signal_filtered))
plt.title('Восстановленный сигнал (от нуля до fs)')

plt.tight_layout()
plt.show()
