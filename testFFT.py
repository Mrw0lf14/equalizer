import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def equalize(signal, start, end, koef):
    if (start == 0):
        window_length = (end - start)*2
        window = 1 + koef * np.hamming(window_length)
        window = window[0:(end - start)]
    else:
        window_length = (end - start)
        window = 1 + koef * np.hamming(window_length)
        window = window[0:(end - start)]
    plt.subplot(3, 1, 1)
    plt.plot(window*1000)
    signal[start:end] *= window
    sample_rate = len(signal)
    signal[sample_rate-end:sample_rate-start] *= window[::-1]
    # signal[start:end] *= koef
    # sample_rate = len(signal)
    # signal[sample_rate-end:sample_rate-start] *= koef
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

def limitfunction(func, maxvalue):
    def limitedfunc(x):
        result = func(x)
        if result > maxvalue:
            return maxvalue
        else:
            return result
    return limitedfunc        
# Функция для преобразования Фурье каждого канала
def fourier_transform(audio_data):
    print("Прямое преобразование Фурье")
    num_channels = audio_data.shape[1]
    transformed_data = []
    packet_size = 16384
    
    for i in range(num_channels):
        channel_transformed = []
        channel_data = audio_data[:, i]  # Выбираем i-ый канал
        for i in range(0, len(channel_data), packet_size):
            signal_packets = channel_data[i:i+packet_size]
            signal_size = len(signal_packets)
            if (signal_size < packet_size):
                pad_length = packet_size - len(signal_packets)
                signal_packets = np.pad(signal_packets, (0, pad_length), mode='constant')  # Дополняем нулями
            channel_fft = FFT(signal_packets)  # Преобразование Фурье
            equalize(channel_fft, 100, 1000, 2)
            channel_ifft = IFFT(channel_fft)
            channel_ifft[0:10] /= 100
            channel_ifft[packet_size-10:packet_size] /= 100
            channel_transformed.extend(channel_ifft.tolist())
        transformed_data.append(np.real(channel_transformed)*2)
    return np.array(transformed_data).T

def limitchannelifft(channelifft, maxvalue):
    def limitedchannelifft(values):
        result = channelifft(values)
        for val in result:
            limitedresult = min(val, max_value) 
        return limitedresult
    return limitedchannelifft

def fourier_transform_mono(audio_data):
    print("Прямое преобразование Фурье")
    packet_size = 32768
    # 32768 16384 8192
    channel_transformed = []
    channel_data = audio_data
    for i in range(0, len(channel_data), packet_size):

        signal_packets = channel_data[i:i+packet_size]
        signal_size = len(signal_packets)
        if (signal_size < packet_size):
            pad_length = packet_size - len(signal_packets)
            signal_packets = np.pad(signal_packets, (0, pad_length), mode='constant')  # Дополняем нулями
        channel_fft = FFT(signal_packets)  # Преобразование Фурье
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.abs(channel_fft))
        plt.title('ДПФ')
        equalize(channel_fft, 10, 1000, -0.9)

        plt.subplot(3, 1, 2)
        plt.plot(np.abs(channel_fft))
        plt.title('ДПФ')
        channel_ifft = IFFT(channel_fft)
        # channel_ifft[0:10] /= 100
        # channel_ifft[packet_size-10:packet_size] /= 100
        plt.subplot(3, 1, 3)
        plt.plot(np.real(channel_ifft))
        plt.title('Выходной сигнал')
        channel_transformed.extend(channel_ifft.tolist())
    return np.array(channel_transformed).T

def normalize_signal(signal):
    max_value = np.max(signal)
    print("normilezed:", max_value)
    normalized_signal = signal / max_value
    print("after normilezed:", np.max(normalized_signal))
    return max_value, normalized_signal

def denormalize_signal(signal, max_value):
    print("denormilezed:", np.max(signal))
    denormalized_signal = signal * max_value
    print("after denormilezed:", np.max(denormalized_signal))
    return denormalized_signal
# # Задаем входной сигнал (в данном примере синусоида с частотой 10 Гц)
# fs = 1000  # Частота дискретизации
# t = np.linspace(0, 1, fs)  # Временной промежуток 1 секунда
# signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 150 * t)  + np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 300 * t)

# # Преобразование Фурье
# fft_signal = np.fft.fft(signal)
# signal2 = zero_pad_to_power_of_two(signal)
# fft_signal2 = FFT(signal2)
# freq_start = 80
# freq_end = 400
# koef = 0.0
# fft_signal[freq_start:freq_end] *= koef
# # fft_signal[fs-freq_end:fs-freq_start] *= koef
# N = len(signal)
# freqs = np.linspace(0, fs-1/fs, N)

# reconstructed_signal = np.fft.ifft(fft_signal)
# reconstructed_signal2 = IFFT(fft_signal2)

sample_rate, audio_data = wavfile.read('in.wav')
max_value, signal = normalize_signal(audio_data)
print("proof normalized signal:", np.max(signal))
# Проверяем, если аудиофайл имеет несколько каналов, разделяем на каналы
if audio_data.ndim > 1:
    transformed_data = fourier_transform(signal)
    temp, transformed_data = normalize_signal(transformed_data)
    print("proof normalized fft:", np.max(transformed_data))
else:
    print("Аудиофайл имеет только один канал.")
    transformed_data = fourier_transform_mono(signal)
    temp, transformed_data = normalize_signal(transformed_data)
    
out_data = denormalize_signal(np.real(transformed_data), max_value)

print("proof denormalized:", np.max(out_data))
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(audio_data)
plt.title('Входной сигнал')
plt.subplot(2, 1, 2)
plt.plot(out_data)
plt.title('Выходной сигнал')

plt.figure()
fft_signal = np.fft.fft(audio_data)
plt.subplot(2, 1, 1)
plt.plot(np.abs(fft_signal))
plt.title('Входной сигнал')
fft_signal = np.fft.fft(out_data)
plt.subplot(2, 1, 2)
plt.plot(np.abs(fft_signal))
plt.title('Выходной сигнал')
# inverse_result = inverse_fourier_transform(transformed_data)
# Сохраняем результат обратного преобразования в файл 'out.wav'
wavfile.write('out.wav', sample_rate, out_data.astype(np.int16))
print("Результат обратного преобразования сохранен в файл 'out.wav'.")
# plt.figure()
# plt.subplot(3, 2, 1)
# plt.plot(t, signal)
# plt.title('Входной сигнал')

# plt.subplot(3, 2, 2)
# plt.plot(np.abs(fft_signal))
# plt.title('Спектр сигнала библиотечная функция')

# plt.subplot(3, 2, 3)
# plt.plot(np.abs(fft_signal2))
# plt.title('Спектр сигнала БПФ')

# plt.subplot(3, 2, 4)
# plt.plot(np.real(reconstructed_signal))
# plt.title('Восстановленный сигнал библиотечная функция')

# plt.subplot(3, 2, 5)
# plt.plot(np.real(reconstructed_signal2))
# plt.title('Восстановленный сигнал БОПФ')

# plt.tight_layout()
plt.show()
