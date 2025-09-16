# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 19:19:48 2025

@author: Admin
"""

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt


def funcion_sen(ff, nn, amp = 1, dc = 0, ph = 0, fs = 2):
    
    ts = 1/fs  # Obtengo tiempo al que se toma cada muestra  [s]
    
    tt = np.arange(nn)*ts 
    xx = amp*np.sin(2*np.pi*ff*tt + ph) + dc
    
    return tt, xx

fs = 1000  #Hz
nn = fs    #muestras
snr = 50   #dB
a0 = np.sqrt(2)  #Volts
realizaciones = 200
fr = np.random.uniform(-2, 2, realizaciones)  #vector de frecuencias random

tt, s_1 = funcion_sen(ff = ((fs/4)+fr)*(fs/nn), nn = nn, amp = a0, fs = fs)


#tienen que ser iguales la potencia del snr y la potencia del ruido
pot_ruido = a0**2 / (2*10**(snr/10))
print(f'Potencia del snr = {pot_ruido:3.1f}')

ruido = np.random.normal(0, np.sqrt(pot_ruido), nn)
var_ruido = np.var(ruido)
print(f'Potencia del ruido = {var_ruido:3.3f}')

x_1 = s_1 + ruido  #modelo de señal

S_1 = (1/nn)*fft(s_1)

R = (1/nn)*fft(ruido)

X_1 = (1/nn)*fft(x_1)


ff_eje = np.arange(nn)*(fs/nn)

plt.figure(figsize=(10,4))
plt.plot(ff_eje, np.log10(2*np.abs(X_1)**2)*10, '-', label="SNR 10 dB")  #Mitad de la potencia en cada delta, debo subir +3dB (multiplico por 2)
# plt.plot(ff_eje, np.log10(np.abs(S_1))*20, '-', label="Señal sin ruido")
# plt.plot(ff_eje, np.log10(np.abs(R))*20, '-', label="ruido")
plt.xlim([0,fs/2])   #limito el eje x con fs esta vez
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Amplitud [dB]')
plt.title("FFT")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


tt = tt.reshape(-1, 1)        # reshape para que las filas sean tiempo

#Repetir tt en columnas 
tt_mat = np.tile(tt, (1, realizaciones))           # (1000x200)

#Repetir fr en filas (mismo valor de frecuencia por columna)
fr_mat = np.tile(fr, (nn, 1))                      # (1000x200)

#matriz de senoidales
xx_mat = a0 * np.sin(2*np.pi*((nn/4)+fr_mat)*(fs/nn)*tt_mat)   #me falta el ruido

#FFT normalizada a lo largo del eje del tiempo (filas)
X_mat = (1/nn) * fft(xx_mat, axis=0)

#FFT de la primera columna
X1 = (1/nn) * fft(xx_mat[:, 0])

plt.plot(ff_eje, np.log10(2*np.abs(X_mat)**2)*10)  
plt.xlim([0,fs/2])   #limito el eje x con fs esta vez
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Amplitud [dB]')
plt.title("FFT")
plt.grid(True)
plt.tight_layout()
plt.show()



