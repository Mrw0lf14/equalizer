import streamlit as st
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import array 

# Эквализация одной полосы
def equalize(signal, freq_start, freq_end, koef):
    signal[freq_start:freq_end] *= koef  # Умножаем значения в заданном интервале на коэффициент
    sample_rate = len(signal)
    print(sample_rate)
    signal[sample_rate-freq_end:sample_rate-freq_start] *= koef
    return signal

# Прямое преобразование Фурье
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

# Обратное преобразование Фурье
def IDFT_slow(X):
    """Compute the Inverse Discrete Fourier Transform of the 1D array X"""
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, X) / N

# Прямое быстрое преобразование Фурье
def FFT(x):
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

# Обратное быстрое преобразование Фурье
def IFFT(X):  
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
        
# Функция для преобразования Фурье каждого канала
def fourier_transform(audio_data, slider_values):
    num_channels = audio_data.shape[1]
    transformed_data = []
    packet_size = 8192
    
    for i in range(num_channels):
        channel_transformed = []
        channel_data = audio_data[:, i]  # Выбираем i-ый канал
        for i in range(0, len(channel_data), packet_size):
            signal_packets = channel_data[i:i+packet_size]
            signal_size = len(signal_packets)
            if (signal_size < packet_size):
                # signal_packets = zero_pad_to_power_of_two(signal_packets)
                pad_length = packet_size - len(signal_packets)
                signal_packets = np.pad(signal_packets, (0, pad_length), mode='constant')  # Дополняем нулями
            channel_fft = FFT(signal_packets)  # Преобразование Фурье
            equalize(channel_fft, 0, 2000, slider_values[0])
            equalize(channel_fft, 2000, 4000, slider_values[1])
            equalize(channel_fft, 4000, 6000, slider_values[2])
            equalize(channel_fft, 6000, 8000, slider_values[3])
            equalize(channel_fft, 8000, 10000, slider_values[4])
            equalize(channel_fft, 10000, 12000, slider_values[5])
            channel_ifft = IFFT(channel_fft)
            channel_transformed.extend(channel_ifft.tolist())
        transformed_data.append(channel_transformed)
    return np.array(transformed_data).T

# Загрузка и отображение аудио файла
def load_audio(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

# Отображение графика
def plot_graph(data):
    fft_data = np.fft.fft(data)
    fft_data = np.abs(fft_data) / len(fft_data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(12, 6))
    plt.plot(fft_data)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform of Audio Signal')
    plt.grid()
    st.pyplot()
   
# Загрузка файла и вывод эквалайзера
uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

if uploaded_file is not None:
    st.write("Проиграть исходное аудио:")
    st.audio(uploaded_file, format='audio/mp3')
    sample_rate, data = load_audio(uploaded_file)

    st.header("Эквалайзер")
    # Создание слайдеров для эквалайзера
    slider_values = []
    for i in range(6):
        # st.sidebar.write(f'Band {i+1}')
        slider_value = st.sidebar.slider(f"Полоса {i+1}", 0.0, 2.0, step=0.01)
        slider_values.append(slider_value)
        # st.write(slider_values)
    
    transformed_signal = fourier_transform(data, slider_values)
    wavfile.write('out.wav', sample_rate, transformed_signal.astype(np.int32))
    out_file = "out.wav"
    st.audio(out_file, format='audio/mp3')
    st.write("АЧХ исходного сигнала:")
    plot_graph(data)
    st.write("АЧХ измененного сигнала:")
    plot_graph(transformed_signal)

# Запуск Streamlit приложения
if __name__ == "__main__":
    st.write("Audio Signal Analyzer")
