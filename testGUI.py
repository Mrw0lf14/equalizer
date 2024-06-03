import streamlit as st
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import array 

# Загрузка и отображение аудио файла
def load_audio(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

# Преобразование Фурье
def fourier_transform(data):
    fft_data = np.fft.fft(data)
    fft_data = np.abs(fft_data) / len(fft_data)
    return fft_data

# Отображение графика
def plot_graph(freqs, fft_data):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, fft_data)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform of Audio Signal')
    plt.grid()
    st.pyplot()

# Загрузка файла и вывод эквалайзера
uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    sample_rate, data = load_audio(uploaded_file)
    freqs = np.fft.fftfreq(len(data), 1/sample_rate)
    fft_data = fourier_transform(data)
    plot_graph(freqs, fft_data)

    st.header("Equalizer")
    # Создание слайдеров для эквалайзера
    for i in range(6):
        # st.sidebar.write(f'Band {i+1}')
        slider_value = st.sidebar.slider(f"Gain Band {i+1}", -10.0, 10.0, step=0.1)
        st.write(slider_value)

# Запуск Streamlit приложения
if __name__ == "__main__":
    st.write("Audio Signal Analyzer")
