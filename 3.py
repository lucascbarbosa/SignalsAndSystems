import numpy as np
from numpy.fft import fft,fftfreq,fftshift
from numpy import absolute,angle, linspace, around
from math import pi, sin, pow,cos,sqrt
import matplotlib.pyplot as plt
import pywt

def Round(x):
    return around(x,decimals=4)

def sinc(x):
    if x == 0:
        return 1
    else:

        return float(sin(pi*x)/(pi*x))

def total_energy(signal):
    s = 0
    for point in signal:
        s+= pow(point,2)
    return(around(s,decimals=2))

def dwt(signal, level):
    if level== 1:
        tend = sub_tend(signal,1)
        flo = sub_float(signal)

    else:
        tend = []
        flo = sub_float(signal)
        tend = sub_tend(signal,1)
        for i in range(2,level+1):
            flo = sub_float(tend) + flo
            tend = sub_tend(tend,i)
        
    return tend,flo

def sub_tend(signal,level):
    a = []
    if level == 1:
        
        for i in range(0,len(signal)-1,2):
            a.append((signal[i]+signal[i+1])/sqrt(2))
        return a
    else:
        return sub_tend(signal,level-1)

def sub_float(signal):
    d = []
    for i in range(0,len(signal)-1,2):
        d.append((signal[i]-signal[i+1])/sqrt(2))
    return d

def get_energy_per_point(signal,tot_en):
    energies =[]
    energy = 0
    for x in signal:
        energy += float(pow(x,2)/tot_en)
        energies.append(energy)
    return energies

comp = 1
To = 5
fs = 300
N  = 1 + To*fs
T = linspace(-To/2,To/2,N)
x = [2*(1+4*cos(6*pi*t))*sinc(2*t) for t in T]
plt.plot(T,x,label='x(t)')
plt.suptitle('X(t)')
plt.xlabel('t')
levels=  [1,2,3,10]
fig,axs = plt.subplots(len(levels))
energies = []
for level in levels:
    fig.suptitle('Haar para diferentes níveis')
    idx = levels.index(level)
    cA,cD= dwt(x,level)
    cA = list(map(Round,cA))
    cD = list(map(Round,cD))
    haar = cA+cD
    comp = total_energy(cA)/(total_energy(haar))
    print('Compressão Haar lvl %i = %f'%(level,comp))
    newT = linspace(-To/2,To/2,len(haar))
    axs[idx].plot(newT,haar)
    axs[idx].set_title('Haar lvl %i'%level)
    energies_haar = get_energy_per_point(haar,total_energy(haar))
    energies.append(energies_haar)

print('\n')

fig,axs = plt.subplots(len(energies))
for en in energies:
    idx = energies.index(en)
    axs[idx].plot(newT,energies[idx][:len(newT)])
    axs[idx].set_title('Eenergia acumulada - Haar lvl %i'%levels[idx])

#Para achar o limiar em que a energia vale 99.99%
limiares = []
for energy in energies:
    idx1 = energies.index(energy)
    for en in energy:
        if en >= 0.9999:
            idx2 = energy.index(en)
            limiar =  newT[idx2]
            limiares.append(limiar)
            break

fig,axs = plt.subplots(len(levels))
energies = []
for level in levels:
    idx = levels.index(level)
    limiar = limiares[idx]
    T = linspace(-To/2,limiar,N)
    x = [2*(1+4*cos(6*pi*t))*sinc(2*t) for t in T]    
    fig.suptitle('Haar no limiar para diferentes níveis')
    cA,cD= dwt(x,level)
    cA = list(map(Round,cA))
    cD = list(map(Round,cD))
    haar = cA+cD
    newT = linspace(-To/2,limiar,len(haar))
    comp = total_energy(cA)/(total_energy(haar))
    print('Compressão Haar (L = %f) lvl %i = %f'%(limiar,level,comp))
    axs[idx].plot(newT,haar)
    axs[idx].set_title('Haar no limiar (L = %f) lvl %i'%(limiar,level))
    energies_haar = get_energy_per_point(haar,total_energy(haar))
    energies.append(energies_haar)

print('\n')


#PARA A DAUB

fig,axs = plt.subplots(3)
cA,cD = pywt.dwt(x,'db1')
daub = cA+cD

comp = float(total_energy(cA)/total_energy(daub))
newT = linspace(-To/2,To/2,len(daub))
print('Taxa de Compressão daub 1: %f'%comp)
plt.suptitle('Daub 1: Sinal e energia acumulada')
axs[0].set_title('Daub 1')
axs[0].plot(newT[:len(daub)],daub)
energy_daub = get_energy_per_point(daub,total_energy(daub))
axs[1].set_title('Energia acumulada')
axs[1].plot(newT[:len(energy_daub)],energy_daub)


#achar limiar
for en in energy_daub:
    if en >= 0.99:
        idx = energy_daub.index(en)
        limiar = newT[idx]
        break

#plot tar x(t) para To/2<=x<=limiar
print(limiar)
T = linspace(-To/2,limiar,N)
x = [2*(1+4*cos(6*pi*t))*sinc(2*t) for t in T]
cA_limiar,cD_limiar = pywt.dwt(x,'db1')
daub_limiar = cA_limiar+cD_limiar
axs[2].set_title('Daub 1 no limiar')
axs[2].plot(T[:len(daub_limiar)],daub_limiar)
comp_limiar = float(total_energy(cA_limiar)/total_energy(daub_limiar))
print('Taxa de compressão Daub 1 no limiar(L = %f) = %f)' %(limiar,comp_limiar))

plt.show()


