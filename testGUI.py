import streamlit as st
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# Эквализация одной полосы
def equalize(signal, start_f, end_f, koef):
    sample_rate = len(signal)
    start = int(np.round(start_f / 44100 * sample_rate))
    end = int(np.round(end_f / 44100 * sample_rate))

    signal[start:end] *= 1 + koef
    sample_rate = len(signal)
    signal[sample_rate-end:sample_rate-start] *= 1 + koef
    return signal

# Прямое преобразование Фурье
def DFT_slow(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

# Обратное преобразование Фурье
def IDFT_slow(X):
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
    if N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

# Обратное быстрое преобразование Фурье
def IFFT(X):  
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    if N <= 32:  # this cutoff should be optimized
        return IDFT_slow(X)
    else:
        X_even = IFFT(X[::2])
        X_odd = IFFT(X[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

# Функция для преобразования Фурье каждого канала
def fourier_transform(audio_data, slider_values):
    num_channels = audio_data.shape[1] if audio_data.ndim > 1 else 1
    transformed_data = []
    packet_size = 16384
    overlap = packet_size // 2
    
    for channel in range(num_channels):
        channel_data = audio_data[:, channel] if num_channels > 1 else audio_data
        transformed_channel = np.zeros_like(channel_data, dtype=np.float32)
        
        for start in range(0, len(channel_data) - packet_size + 1, overlap):
            signal_packet = channel_data[start:start+packet_size]
            signal_packet = signal_packet.copy()  # Ensure the array is writable
            
            # windowed_packet = signal_packet * np.hanning(packet_size)
            channel_fft = FFT(signal_packet)
            
            equalize(channel_fft, 0, 3000, slider_values[0])
            equalize(channel_fft, 3000, 6000, slider_values[1])
            equalize(channel_fft, 6000, 9000, slider_values[2])
            equalize(channel_fft, 9000, 12000, slider_values[3])
            equalize(channel_fft, 12000, 15000, slider_values[4])
            equalize(channel_fft, 15000, 18000, slider_values[5])
            
            ifft_result = IFFT(channel_fft).real
            transformed_channel[start:start+packet_size] += ifft_result * np.hanning(packet_size)
        
        transformed_data.append(transformed_channel)
    
    if num_channels > 1:
        return np.array(transformed_data).T
    else:
        return np.array(transformed_data).flatten()

def normalize_signal(signal):
    max_value = np.max(np.abs(signal))
    normalized_signal = signal / max_value
    return max_value, normalized_signal

def denormalize_signal(signal, max_value):
    denormalized_signal = signal * max_value
    return denormalized_signal

# Отображение графика
def plot_fft(data):
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
    
def plot_raw(data):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.grid()
    st.pyplot()
    
# Загрузка файла и вывод эквалайзера
uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

if uploaded_file is not None:
    st.write("Проиграть исходное аудио:")
    st.audio(uploaded_file, format='audio/wav')
    sample_rate, audio_data = wavfile.read(uploaded_file)
    max_value, signal = normalize_signal(audio_data)
    st.header("Эквалайзер")
    # Создание слайдеров для эквалайзера
    slider_values = []
    for i in range(6):
        slider_value = st.sidebar.slider(f"Полоса {i*3000} - {(i+1)*3000}", -1.0, 2.0, step=0.01)
        slider_values.append(slider_value)

    transformed_signal = fourier_transform(signal, slider_values)
    temp, transformed_signal = normalize_signal(transformed_signal)
    out_data = denormalize_signal(np.real(transformed_signal), max_value)
    out_file = "out_" + uploaded_file.name
    wavfile.write(out_file, sample_rate, out_data.astype(np.int16))

    st.audio(out_file, format='audio/wav')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("Исходный сигнал")
    plot_raw(audio_data)
    st.write("Выходной сигнал")
    plot_raw(out_data)
    st.write("АЧХ исходного сигнала:")
    plot_fft(audio_data)
    st.write("АЧХ измененного сигнала:")
    plot_fft(out_data)

# Запуск Streamlit приложения
if __name__ == "__main__":
    st.write("Audio Signal Analyzer")
