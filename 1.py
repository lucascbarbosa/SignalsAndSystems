import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.stats import chisquare

T = 1
i = 0
t_max = 8

def iterate(T,k):
    if k == 0:
        return 0
    else:
        return float(iterate(T,k-1)*(1-T*(1+math.sin(k-T)))+T)


last_curve = [0 for i in range(t_max+1)]

fig, axs = plt.subplots(2)

for i in range(2):
    i+=1
    x = []
    y = []
    for k in range(t_max*i+1):
        res = iterate(1,k)
        x.append(k)
        y.append(res)
    i -= 1
    print(y)
    axs[i].plot(x,y)
    axs[i].set_title('Y(k) vs k (T=1/%i)'%(i+1))

plt.savefig('plots/1b-d.png')
plt.show()



for f in range(3,21):
    y = []
    x = []
    for k in range(t_max*f+1):
        x.append(k)
        res = iterate(float(1/f),k)
        y.append(res)
    # a nova curva(y) será sempre maior que a anterior (last_curve), logo devemos selecionar uma parte do dominio
    # de last_curve que existe em y tiramos o primeiro elemento pois ele vale 0, que não é possivel de ser 
    # utilizado, já que daria uma indeterminação
    plt.suptitle('Y(k) vs k para cada valor de T')
    plt.ylabel('y(k)')
    plt.xlabel('k')
    plt.plot(x,y, label = 'T = 1/%i R2 = %f'%(f,chisquare(y[1:4],last_curve[1:4])[1]))
    last_curve = y


plt.savefig('plots/1f.png')
plt.legend(loc='best')
plt.show()




