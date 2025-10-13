# -*- coding: utf-8 -*-
"""
Comparación de ventanas en el dominio de la frecuencia (eje en múltiplos de Δf)
Autor: [Tu nombre]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from numpy.fft import fft, fftshift

# ================== PARÁMETROS ==================
NN = 1000         # número de puntos de cada ventana
NFFT = 9*NN      # longitud de la FFT para buena resolución
FS = 1000         # asumo Fs = 1 Hz para que Δf = 1/NFFT
DELTA_F = FS / NN
tt = np.arange(NN) / FS  # vector de tiempo


# ================== DEFINICIÓN DE VENTANAS ==================
ventanas = {
    "Rectangular": windows.boxcar(NN),
    "Hamming": windows.hamming(NN),
    "Hann": windows.hann(NN),
    "Blackman Harris": windows.blackmanharris(NN),
    "Flattop": windows.flattop(NN)
}

# ================== ANÁLISIS ==================
plt.figure(figsize=(10, 5))

for nombre, ventana in ventanas.items():
    # FFT centrada
    W = fft(ventana, NFFT)
    W = fftshift(W)
    W_db = 20 * np.log10(np.abs(W) / np.max(np.abs(W)) + 1e-12)

    # eje en múltiplos de Δf
    k = np.arange(-NFFT//2, NFFT//2)  # enteros
    plt.plot(k, W_db, label=nombre)

# ================== GRÁFICO ==================
plt.title("Ventanas")
plt.xlabel("Frecuencia [múltiplos de Δf]")
plt.ylabel("|Ventana (f)| [dB]")
plt.legend()
plt.ylim([-80, 5])
plt.xlim(-150,150)
plt.grid(True)
plt.show()



# ================== SEÑAL ==================
f0 = 50   # frecuencia de la senoidal
x = np.sin(2*np.pi*f0*tt) + 0.2*np.random.randn(NN)  # señal + ruido

# ================== VENTANAS ==================
ventanas = {
    "Rectangular": windows.boxcar(NN),
    "Hamming": windows.hamming(NN),
    "Hann": windows.hann(NN),
    "Blackman Harris": windows.blackmanharris(NN),
    "Flattop": windows.flattop(NN)
}

# ================== ESTIMACIÓN ESPECTRAL ==================
plt.figure(figsize=(12, 6))

for nombre, w in ventanas.items():
    xw = x * w  # aplicar ventana
    X = fft(xw, NFFT)
    X = fftshift(X)
    X_db = 20*np.log10(np.abs(X)/np.max(np.abs(X)) + 1e-12)
    
    # eje en múltiplos de Δf
    k = np.arange(-NFFT//2, NFFT//2)
    plt.plot(k, X_db, label=nombre)

plt.title("Estimación espectral con diferentes ventanas")
plt.xlabel("Frecuencia [múltiplos de Δf]")
plt.ylabel("Magnitud [dB]")
plt.ylim([-80, 5])
plt.xlim(-150, 150)  # zoom
plt.grid(True)
plt.legend()
plt.show()


