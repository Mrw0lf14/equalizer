import numpy as np

# Задаем входной сигнал (в данном примере синусоида с частотой 10 Гц)
fs = 1000  # Частота дискретизации
t = np.linspace(0, 1, fs)  # Временной промежуток 1 секунда
signal = np.sin(2 * np.pi * 100 * t) + 1/2*np.sin(2 * np.pi * 200 * t)  # 10 Гц синусоида

# Преобразование Фурье
fft_signal = np.fft.fft(signal)

freqs = np.fft.fftfreq(len(signal), 1/fs)

# Обратное преобразование Фурье
reconstructed_signal = np.fft.ifft(fft_signal)
#print(np.abs(fft_signal))
# Оставляем значения от нуля до частоты дискретизации
fft_signal[:int(fs/2)] = 0

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
plt.plot(t, np.real(reconstructed_signal))
plt.title('Восстановленный сигнал (полное преобразование)')

plt.subplot(2, 2, 4)
plt.plot(t, np.real(reconstructed_signal_filtered))
plt.title('Восстановленный сигнал (от нуля до fs)')

plt.tight_layout()
plt.show()
