# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:40:56 2025

@author: Admin
"""

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

fs = 1000
ts = 1/fs
nn = 1000
tt = np.arange(nn)*ts


#FFT de señal senoidal 
x1 = np.sin(tt*np.pi*2*fs/4)
X1 = fft(x1)


#FFT de señal adyacente

x2 = np.sin(2* np.pi* (fs/ 4 + fs/nn)*tt)
X2 = fft(x2)


#FFT de señal en el medio de ambas

x3 = np.sin(2*np.pi*((fs/4)+(fs/(2*nn)))*tt)
X3 = fft(x3)    #Además del módulo puedo calcular su fase con np.angle(X3)

# plt.figure(figsize=(10,4))
# #Al plotear así tengo en función de las muestras
# plt.plot(np.abs(X1), 'x', label="Transformada 1")
# plt.plot(np.abs(X2), 'o', label="Transformada 2(adyacente)")
# plt.plot(np.abs(X3), 'P', label="Transformada 3(punto medio)")
# plt.xlim([0,nn/2])   #limito el eje x
# plt.xlabel("Muestras")
# plt.ylabel('Amplitud')
# plt.title("FFT")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#Al plotear así tengo en función de la frecuencia en Hz
ff_eje = np.arange(nn)*(fs/nn)   #Utilizo para visualización espectral como eje x

plt.figure(figsize=(10,4))
# plt.plot(ff_eje, np.abs(X1), 'x', label="Transformada 1")
# plt.plot(ff_eje, np.abs(X2), 'x', label="Transformada 2(adyacente)")
# plt.plot(ff_eje, np.abs(X3), 'x', label="Transformada 3(punto medio)")
plt.plot(ff_eje, np.log10(np.abs(X1))*20, 'x', label="Transformada 1 en dB")
plt.plot(ff_eje, np.log10(np.abs(X2))*20, 'o', label="Transformada 2(adyacente) en dB")
plt.plot(ff_eje, np.log10(np.abs(X3))*20, '^', label="Transformada 3(punto medio) en dB")
plt.xlim([0,fs/2])   #limito el eje x con fs esta vez
# plt.xlabel("Muestras")
plt.xlabel("Frecuencia [Hz]")
# plt.ylabel('Amplitud')
plt.ylabel('Amplitud [dB]')
plt.title("FFT")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()