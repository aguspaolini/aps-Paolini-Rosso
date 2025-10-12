# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:58:32 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import fft, fftfreq

def crear_senal(amp, fs, snr_db, N):
    t = np.arange(N)
    omega_0 = fs / 4
    fr = np.random.uniform(-2, 2)
    omega_1 = omega_0 + fr * fs / N
    senal_pura = amp * np.sin(2*np.pi*omega_1 * t / fs)  #frecuencia → 2πf/fs * n
    
    pot_senal = np.mean(senal_pura ** 2)
    snr_lineal = 10 ** (snr_db / 10)
    pot_ruido = pot_senal / snr_lineal
    ruido = np.random.normal(0, np.sqrt(pot_ruido), len(t))
    
    senal_completa = senal_pura + ruido
    
    return senal_completa, senal_pura, omega_1

def estimadores(senal, ventana, Nfft, fs, omega_0):
    # Aplicar ventana
    senal_win = senal * ventana
    espectro = fft(senal_win, Nfft)
    freqs = fftfreq(Nfft, 1/fs)

    mag = np.abs(espectro)

    # índice más cercano a ω0
    idx_omega0 = np.argmin(np.abs(freqs - omega_0))
    amp_est = np.abs(espectro[idx_omega0])

    # estimador de frecuencia: máximo
    idx_max = np.argmax(mag)
    freq_est = freqs[idx_max]

    return amp_est, freq_est, freqs, mag


# Parámetros
amp = np.sqrt(2)
fs = 1000
N = 1000
Nfft = 2**int(np.ceil(np.log2(N)))
snr_db = 10   # probar también con 3 dB

# Ventanas
ventanas = {
    "Rectangular": np.ones(N),
    "Flattop": get_window("flattop", N),
    "Blackman-Harris": get_window("blackmanharris", N),
    "Hamming": get_window("hamming", N)
}


# Un solo experimento
senal, senal_pura, omega_1_real = crear_senal(amp, fs, snr_db, N)

plt.figure(figsize=(12, 6))

for i, (nombre, ventana) in enumerate(ventanas.items(), 1):
    amp_est, freq_est, freqs, mag = estimadores(senal, ventana, Nfft, fs, fs/4)
    
    print(f"Ventana {nombre}:")
    print(f"  Amplitud estimada = {amp_est:.3f}")
    print(f"  Frecuencia estimada = {freq_est:.2f} Hz")
    print("-"*40)
    
    plt.plot(freqs[:Nfft//2], mag[:Nfft//2], label=nombre)

plt.title(f"Espectro con distintas ventanas (SNR = {snr_db} dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()
plt.show()
