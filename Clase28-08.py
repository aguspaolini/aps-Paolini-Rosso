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
import matplotlib.pyplot as plt
from scipy import signal

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
X_1_abs = np.abs(X_1)

d_e_pot = (1/nn)*(np.abs(X_1))**2
print('Densidad espectral de potencia =', d_e_pot)
print('Densidad espectral de potencia en dB =', 10*np.log10(d_e_pot))

p_der = (1/nn)*np.sum((np.abs(X_1))**2)  #en frecuencia
p_izq = np.sum((np.abs(x_1))**2)         #en energía

print('Parseval -->', p_izq, ' = ', p_der)

#%%----------------------- Zero padding -----------------------

ff_eje = np.arange(nn)*(fs/nn)   #Utilizo para visualización espectral como eje x
Npad = 9 * nn
x_2 = np.zeros(Npad)
x_2[:nn] = x_1
X_2 = fft(x_2)
X_2_abs = np.abs(X_2)
ff_eje_2 = np.arange(Npad) * fs /Npad

plt.figure(4)
plt.plot(ff_eje, X_1_abs,'x', label="Transformada 1 en N/4")
plt.plot(ff_eje_2, X_2_abs, label="Transformada 1 en N/4 con zero-padding")
#plt.xlim(240,260)
plt.title("FFT")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Modulo de la señal')
plt.legend()
plt.grid(True)
plt.show()

# #%% -------------------- Agrego ruido ---------------------

np.random.seed(0)  # reproducibilidad
ruido = np.random.normal(0, 0.5, nn)   # ruido gaussiano con sigma=0.5
x_ruidosa = x_1 + ruido


X_ruidosa = fft(x_ruidosa)
X_ruidosa_abs = np.abs(X_ruidosa)

# ------------------- CALCULO DE SNR -------------------
P_signal = np.mean(x_1**2)              # potencia de la señal pura
P_noise = np.mean(ruido**2)             # potencia del ruido
SNR = 10*np.log10(P_signal/P_noise)     # en dB

print(f"SNR = {SNR:.2f} dB")


x_3 = np.zeros(Npad)
x_3[:nn] = x_ruidosa
X_3 = fft(x_3)
X_3_abs = np.abs(X_3)


plt.figure(4)
plt.plot(ff_eje, X_1_abs,'x', label="Transformada con ruido")
plt.plot(ff_eje_2, X_3_abs, label="Transformada ruidosa con zero-padding")
plt.xlim(240,260)
plt.title("FFT")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo de la señal')
plt.legend()
plt.grid(True)
plt.show()


#%% ----------------- Generar ventanas -----------------------

# Ventana Hamming
ventana = signal.windows.hamming(nn)
plt.figure()
plt.plot(ventana)
plt.title("Ventana Hamming")
plt.ylabel("Amplitud")
plt.xlabel("Muestras")
plt.grid(True)

# Señal multiplicada por la ventana
x_ventana = x_1 * ventana
X_VENTANA = fft(x_ventana)

plt.figure()
plt.plot(ff_eje, X_VENTANA, 'b', label='Ventana Hamming en señal')
plt.title("Ventana Hamming")
plt.ylabel("Amplitud")
plt.xlabel("Muestras")
plt.grid(True)

