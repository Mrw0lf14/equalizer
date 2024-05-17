import numpy as np
import numpy as np

def equalize(signal, freq_start, freq_end, koef):

    signal[freq_start:freq_end] *= koef  # Умножаем значения в заданном интервале на коэффициент
    
    return signal

def zero_pad_to_power_of_two(signal):
    current_length = len(signal)
    next_power_of_two = 2**int(np.ceil(np.log2(current_length)))  # Находим ближайшую степень двойки
    pad_length = next_power_of_two - current_length
    padded_signal = np.pad(signal, (0, pad_length), mode='constant')  # Дополняем нулями
    return padded_signal

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def IDFT_slow(X):
    """Compute the Inverse Discrete Fourier Transform of the 1D array X"""
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, X) / N
 
def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

def IFFT(X):  # Обратное преобразование Фурье
    """A recursive implementation of the 1D Cooley-Tukey IFFT"""
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of X must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return IDFT_slow(X)
    else:
        X_even = IFFT(X[::2])
        X_odd = IFFT(X[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

# Задаем входной сигнал (в данном примере синусоида с частотой 10 Гц)
fs = 1000  # Частота дискретизации
t = np.linspace(0, 1, fs)  # Временной промежуток 1 секунда
signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 150 * t)  + np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 300 * t)

# Преобразование Фурье
fft_signal = np.fft.fft(signal)
signal2 = zero_pad_to_power_of_two(signal)
fft_signal2 = FFT(signal2)
freq_start = 80
freq_end = 400
koef = 0.0
fft_signal[freq_start:freq_end] *= koef
fft_signal[fs-freq_end:fs-freq_start] *= koef
N = len(signal)
freqs = np.linspace(0, fs-1/fs, N)

reconstructed_signal = np.fft.ifft(fft_signal)
reconstructed_signal2 = IFFT(fft_signal2)

print(len(reconstructed_signal))
print(len(reconstructed_signal2))

import matplotlib.pyplot as plt

plt.subplot(3, 2, 1)
plt.plot(t, signal)
plt.title('Входной сигнал')

plt.subplot(3, 2, 2)
plt.plot(np.abs(fft_signal))
plt.title('Спектр сигнала библиотечная функция')

plt.subplot(3, 2, 3)
plt.plot(np.abs(fft_signal2))
plt.title('Спектр сигнала БПФ')

plt.subplot(3, 2, 4)
plt.plot(t, np.real(reconstructed_signal))
plt.title('Восстановленный сигнал библиотечная функция')

plt.subplot(3, 2, 5)
plt.plot(np.real(reconstructed_signal2))
plt.title('Восстановленный сигнал БОПФ')

plt.tight_layout()
plt.show()
