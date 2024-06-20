import numpy as np
import matplotlib.pyplot as plt

def apply_window(signal, window_function=np.hanning):
    """
    Применяет оконную функцию к сигналу.

    Параметры:
    signal (numpy.ndarray): Массив сигнала.
    window_function (callable): Оконная функция, по умолчанию Ханнинг.

    Возвращает:
    numpy.ndarray: Сигнал с примененной оконной функцией.
    """
    window = window_function(len(signal))
    return signal * window

def fft_with_window(signal):
    """
    Применяет преобразование Фурье с использованием оконной функции и zero-padding.

    Параметры:
    signal (numpy.ndarray): Массив сигнала.

    Возвращает:
    numpy.ndarray: Спектр сигнала.
    """
    # Применение оконной функции
    # windowed_signal = apply_window(signal)
    
    # # Применение zero-padding
    # padded_signal = np.pad(windowed_signal, (0, len(windowed_signal)), mode='constant')
    
    # Прямое преобразование Фурье
    spectrum = np.fft.fft(signal)
    
    return spectrum

def ifft_with_window(spectrum, original_length):
    """
    Применяет обратное преобразование Фурье с использованием оконной функции и обрезкой.

    Параметры:
    spectrum (numpy.ndarray): Спектр сигнала.
    original_length (int): Оригинальная длина сигнала.

    Возвращает:
    numpy.ndarray: Восстановленный сигнал.
    """
    # Обратное преобразование Фурье
    reconstructed_signal = np.fft.ifft(spectrum)
    
    # # Обрезка сигнала до оригинальной длины
    # reconstructed_signal = reconstructed_signal[:original_length]
    
    # # Применение оконной функции
    # reconstructed_signal = apply_window(reconstructed_signal)
    
    return reconstructed_signal

# Пример использования
if __name__ == "__main__":
    # Пример сигнала
    t = np.linspace(0, 1, 500, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    
    # Прямое преобразование Фурье с оконной функцией и zero-padding
    spectrum = fft_with_window(signal)
    
    # Обратное преобразование Фурье
    reconstructed_signal = ifft_with_window(spectrum, len(signal))
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.title("Оригинальный сигнал")
    plt.plot(t, signal)
    
    plt.subplot(2, 1, 2)
    plt.title("Восстановленный сигнал")
    plt.plot(t, reconstructed_signal.real)  # Используем только реальную часть сигнала
    
    plt.tight_layout()
    plt.show()
