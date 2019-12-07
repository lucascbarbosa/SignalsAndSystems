import numpy as np
from numpy.fft import fft,fftfreq,fftshift
from numpy import absolute,angle
from math import pi, sin, pow
import matplotlib.pyplot as plt
T = np.linspace(-0.5,0.5,14)
def sinc(x):
    if x == 0:
        return 1
    else:

        return float(sin(pi*x)/(pi*x))

x = [(10*(sinc(40*t)+pow(sinc(10*t),2))) for t in T]
plt.plot(T,x)
plt.savefig('plots/2a.png')
plt.suptitle('x(t) x t')
plt.ylabel('x(t)')
plt.xlabel('t')
plt.show()
To = 1

def Fft(signal,ts):
    signal = np.array(signal)
    fs = int(1/To)
    Fo = int((signal.size-1)*fs)
    freq = fftshift(fftfreq(signal.size,d=fs))
    result = fftshift(fft(signal))
    print(fs,Fo,freq)
    amps = list(map(amp,result))
    phases = list(map(phase,result))
    fig, axs = plt.subplots(2)
    plt.suptitle("Resultado da FFT")
    axs[0].plot(freq,amps)
    axs[0].set_title('Amplitude')
    axs[1].plot(freq,phases)
    axs[1].set_title('Fase')
    plt.savefig('plots/fft.png')
    plt.show()
    print(amps)
    print(phases)
    return freq,result

def amp(x):

    return absolute(x)

def phase(x):
    
    return angle(x)
x = np.array(x)
"""Fft(x,0.00125)"""