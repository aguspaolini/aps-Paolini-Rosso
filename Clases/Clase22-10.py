# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 18:35:10 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Plantilla de diseño
fs=1000 # Hz
WP = [0.8, 35] # comienzo y fin de la banda de paso
WS = [0.1, 40] # comienzo y fin de la banda de stop (corresponde a la atenuacion minima y maxima) -- mas grande que la de paso


alpha_p = 1  # atenuacion
alpha_s = 40

# Aproximaciones de modulo
# f_aprox = 'butter'
#f_aprox = 'cheby1'
f_aprox = 'cheby2'
#f_aprox = 'ellip'

# Aproximacion de fase
# f_aprox = 'bessel'

# Diseño del filtro Butterworth analogico
mi_sos = signal.iirdesign(wp=WP, ws=WS, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)


#%%

# Respuesta en frecuencia
# w, h = signal.freqs(b, a, worN=np.logspace(1, 6, 1000)) # espacio logaritmicamente espaciado
#w, h = signal.freqs(mi_sos) # calcula la respuesta en frecuencia del filtro
#w, h = signal.freqz_sos(sos=mi_sos, fs=fs) # agregamos fs=fs para que el w salga en hz
w, h = signal.freqz_sos(sos=mi_sos, fs=fs, worN=np.logspace(-2,1.9,1000))

# --- Cálculo de fase y retardo de grupo ---
phase = np.unwrap(np.angle(h))
# Retardo de grupo = -dφ/dω
w_rad = w / (fs/2) *np.pi
gd = -np.diff(phase) / np.diff(w_rad)

# --- Polos y ceros ---
z, p, k = signal.sos2zpk(mi_sos)

# --- Gráficas ---
plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.plot(w, 20*np.log10(abs(h)),label=f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(2,2,2)
plt.plot(w, np.degrees(phase))
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(2,2,3)
plt.plot(w[:-1], gd)
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()