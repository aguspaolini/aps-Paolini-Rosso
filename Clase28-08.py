# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 21:37:10 2025

@author: Admin
"""

#1) Identificar senoidal con varianza = 1
#2) Calcular densidad espectral de potencia 
#3) Verificar Parseval

import numpy as np
from numpy.fft import fft

def funcion_sen(ff, nn, amp = 1, dc = 0, ph = 0, fs = 2):
    
    ts = 1/fs  # Obtengo tiempo al que se toma cada muestra  [s]
    
    tt = np.arange(nn)*ts 
    xx = amp*np.sin(2*np.pi*ff*tt + ph) + dc
    
    return tt, xx

fs = 1000
nn = fs

tt, x_1 = funcion_sen(ff = fs/4, nn = fs, amp = np.sqrt(2), fs = fs)

var_1 = np.var(x_1)
media_1 = np.mean(x_1)
desv_1 = np.std(x_1)

print('Varianza =', var_1)
print('Media =', media_1)
print('Desvío estándar =', desv_1)
print('\n')

X_1 = fft(x_1)

d_e_pot = (np.abs(X_1))**2
print('Densidad espectral de potencia =', d_e_pot)
print('Densidad espectral de potencia en dB =', 10*np.log10(d_e_pot))

p_der = (1/nn)*np.sum((np.abs(X_1))**2)  #en frecuencia
p_izq = np.sum((np.abs(x_1))**2)         #en energía

print('Parseval -->', p_izq, ' = ', p_der)