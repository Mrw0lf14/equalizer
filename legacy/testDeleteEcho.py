import numpy as np
import matplotlib.pyplot as plt
# Задаем входной сигнал (в данном примере синусоида с частотой 10 Гц)
fs = 1000  # Частота дискретизации
t = np.linspace(0, 1, fs)  # Временной промежуток 1 секунда
signal = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 0.5 * t) + np.sin(2 * np.pi * 20 * t) 
plt.subplot(2, 2, 1)
plt.plot(t, signal)
plt.title('Входной сигнал')

N = len(signal)
signal2 = np.zeros(N)
delay = 100
# echo
for i in range(delay, len(signal)):
    signal2[i] = signal[i] + 0.5*signal[i-delay]

plt.subplot(2, 2, 2)
plt.plot(t, signal2)
plt.title('ЭХО')

autocorr = np.zeros(N)
for d in range(N):
    for i in range(N-d):
        autocorr[i] += signal2[i] * signal2[i + d]
        
signal3 = np.zeros(N)
for i in range(delay, len(signal)):
    signal3[i] = signal2[i] - 0.5*signal2[i-delay]




plt.subplot(2, 2, 3)
plt.plot(t, autocorr)
plt.title('АВТОКОР')

plt.subplot(2, 2, 4)
plt.plot(t, signal3)
plt.title('-ЭХО')

plt.tight_layout()
plt.show()
