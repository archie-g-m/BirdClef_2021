import numpy as np
from scipy.fftpack import fft
import os
import librosa
import matplotlib.pyplot as plt


def importData():
    data = np.array([], dtype=object)
    numfiles = len(os.listdir('birdclef-2021/train_short_audio/acafly'))
    countFiles = 0
    for filename in os.listdir('birdclef-2021/train_short_audio/acafly'):
        print(filename)
        # countFiles+=1
        # print(str(round(100*countFiles/numfiles, 2))+"%", end='\r')
        y, sr = librosa.load('birdclef-2021/train_short_audio/banana/XC112602.ogg')
        # print(y.shape)
        print(y.shape[0]/sr)
        # Plot the audio signal in time
        time = np.linspace(0, y.shape[0]/sr, y.shape[0])
        plt.figure(1)
        plt.plot(time, y)
        plt.title('Audio signal in time', size=16)

        # spectrum
        n = y.shape[0]
        AudioFreq = fft(y)
        AudioFreq = AudioFreq[0:int(np.ceil((n + 1) / 2.0))]
        MagFreq = np.abs(AudioFreq)
        MagFreq = MagFreq / float(n)

        # power spectrum
        MagFreq = MagFreq**2
        if n % 2 > 0:  # ffte odd
            MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
        else:  # fft even
            MagFreq[1:len(MagFreq) - 1] = MagFreq[1:len(MagFreq) - 1] * 2
        freqAxis = np.arange(0, int(np.ceil((n + 1) / 2.0)), 1.0) * (sr / n)
        plt.figure(2)
        plt.plot(freqAxis / 1000.0, 10 * np.log10(MagFreq))  # Power spectrum
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Power spectrum (dB)')

        plt.show()
        break

        # print(y)
    # print(type(y).__name__)
    # print(y.shape)


if __name__ == '__main__':
    importData()
