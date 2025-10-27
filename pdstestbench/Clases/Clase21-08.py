# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 20:55:24 2025

@author: Admin
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
# #%%   Ejercicio pulso y autocorrelación
# N = 10

# pulso = np.zeros(N)
# pulso[0:3] = 1

# delta = np.zeros(N)
# delta[5] = 1     #implica un desplazamiento

# Rxx = sig.correlate(pulso, delta)
# conv = sig.convolve(pulso,delta)

# plt.plot(pulso, 'x:')
# plt.plot(delta, 'x:')
# plt.plot(Rxx, 'o:')
# plt.plot(conv, 'o:')
# plt.show()


#%%  Ejercicio DFT

fs = 1000  # Frecuencia de muestreo [Hz]
ts = 1/fs     # Tiempo entre muestras [s]
nn = 1000      # Cantidad de muestras
tt = np.arange(nn)*ts   # Vector de tiempo


N = 20
n = np.arange(N) % 8

x = 4 + 3*np.sin(n*np.pi/2)
print(x)

transf = np.zeros(N, dtype = complex)

for k in range(N):
    for t in range(N):
        transf += x*np.exp(1j*2*np.pi*k*t/N)

print(transf)
    

        
