from scipy.fftpack import fft
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("start")
    fs, y = wavfile.read('uwu.wav')
    T = 1/fs
    N = len(y)
    x = np.linspace(0.0, N*T, N)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()