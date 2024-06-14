import scipy.signal
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Коэффициенты фильтрации
numerator = [0.000000000651703380907738176811535387898,
             0.000000006517033809077381354525047602467,
             0.000000029326652140848215268182101658073,
             0.000000078204405708928582871745471653824,
             0.00000013685770999062501340810967496997,
             0.000000164229251988750021383687530303341,
             0.00000013685770999062501340810967496997,
             0.000000078204405708928582871745471653824,
             0.000000029326652140848218576904551870184,
             0.000000006517033809077381354525047602467,
             0.000000000651703380907738176811535387898]

denominator = [1,
               -9.165581806097085504347887763287872076035,
               38.22790702081984903770717210136353969574,
               -95.517864555945209303899900987744331359863,
               158.303430496296755336516071110963821411133,
               -181.802048376680659202975220978260040283203,
               146.504677699063734053197549656033515930176,
               -81.793721709797651442386268172413110733032,
               30.277068342297571490462360088713467121124,
               -6.709973904086417917369544738903641700745,
               0.676107542901690727887853427091613411903]

# Чтение аудиофайла
audio_data, sample_rate = sf.read("uwu.wav")

# Применение фильтра
filtered_audio = scipy.signal.lfilter(numerator, denominator, audio_data)

# Сохранение результата в аудиофайл
sf.write("output_audio.wav", filtered_audio*100000, sample_rate)
fft_signal = np.fft.fft(audio_data)
plt.subplot(2, 1, 1)
plt.plot(np.abs(fft_signal))
plt.title('Входной сигнал')

fft_signal = np.fft.fft(filtered_audio)
plt.subplot(2, 1, 2)
plt.plot(np.abs(fft_signal))
plt.title('Спектр сигнала библиотечная функция')
plt.show()
