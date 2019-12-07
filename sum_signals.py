import numpy as np
from numpy.fft import fft,fftfreq,fftshift
from numpy import absolute,angle
import math
import matplotlib.pyplot as plt

def merge(list1, list2): 
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 

def demerge(tuples):
    return list(map(list,zip(*tuples)))

def round(x):
    return np.around(x,decimals = 2)

def amp(x):

    return absolute(x)

def phase(x):
    
    return angle(x)

class PlotPulses():
    def __init__(self, window,fs):
        self.window = window
        self.fs = fs
        self.samples = window*fs+1
        self.lista_x =  np.linspace((-1)*self.window/2,self.window/2,self.samples)


    def pulseStep(self,delta,alfa):
        lista_y = []
        for x in self.lista_x:
            x = np.around(x+alfa,decimals=1)            
            
            if x >= -delta and x <= delta:
                y = delta
                
            else:
                y = 0
            lista_y.append(y)
        tuples = merge(self.lista_x,lista_y)
        plt.plot(self.lista_x,lista_y)
        plt.suptitle('Pulso Retangular')
        plt.savefig('plots/pulso_ret.png')
        plt.show()
        return tuples

    def pulseTriang(self,delta,alfa): 
        lista_y = []
        offset = float(math.pi*delta/2)
        slope = float(offset/delta)
        for x in self.lista_x:
            x = np.around(x+alfa,decimals=1)
            if x < -delta or x > delta:
                y = 0
            elif x < 0:
                y = offset +slope*x
            elif x == 0:
                y = offset
            elif x > 0:
                y = offset -slope*x
            
            lista_y.append(y)
        tuples = merge(self.lista_x,lista_y)
        plt.plot(self.lista_x,lista_y)
        plt.savefig('plots/pulso_triang.png')
        plt.suptitle('Pulso Triangular')
        plt.show()
        return tuples

    def pulseSemiCirc(self,delta,alfa):
        lista_y = []
        for x in self.lista_x:
            x = np.around(x+alfa,decimals=1)            
            if x < -delta  or x > delta:
                y = 0
            elif x >= -delta and x <=delta:
                y = math.sqrt(delta**2-math.pow(x,2))
            
            lista_y.append(y) #x²+y²=r², r = delta
        tuples = merge(self.lista_x,lista_y)     
        plt.plot(self.lista_x,lista_y)
        plt.suptitle('Pulso Semicircular')
        plt.savefig('plots/pulso_semicirc.png')
        plt.show()
        return tuples

    def sumSignals(self,signalsWeights):
        Y = np.zeros(self.samples)
        for weight,signal in signalsWeights:
            x,y = demerge(signal)
            x = np.around(x,decimals=1)
            for tup in signal:
                x,y = tup
                idx = np.where(self.lista_x == x)
                Y[idx] += weight*y
        plt.plot(self.lista_x,Y)
        plt.suptitle('Soma dos sinais')
        plt.savefig('plots/soma_sinais.png')
        plt.show()
        return Y
    
    def fft(signal,ts):
        signal = np.array(signal)
        freqs = signal.size
        freq = fftshift(fftfreq(freqs,d=ts))
        result = fftshift(fft(signal))
        amps = list(map(amp,result))
        phases = list(map(phase,result))
        fig, axs = plt.subplots(2)
        plt.suptitle("Resultado da FFT")
        axs[0].plot(freq,amps)
        axs[0].set_title('Amplitude')
        axs[1].plot(freq,phases)
        axs[1].set_title('Fase')
        plt.plot(self.lista_x,)
        plt.savefig('plots/fft.png')
        plt.show()
        print(amps)
        print(phases)
        return freq,result


plot = PlotPulses(10,10)
signal1 = plot.pulseTriang(2,-2)
signal2 = plot.pulseSemiCirc(2,2)
signal3 = plot.pulseStep(3,0)
listas = [(1,signal1),(1,signal2),(1,signal3)]
signal_result = plot.sumSignals(listas)
plot.fft(signal_result)