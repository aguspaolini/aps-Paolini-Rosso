# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:32:33 2025

@author: Admin
"""

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import get_window


# ==============================
# Parámetros generales
# ==============================
fs = 1000          # Hz
nn = fs            # muestras
realizaciones = 200
a0 = np.sqrt(2)    # amplitud calibrada (potencia de 1 W)
snrs = [3, 10]     # SNRs en dB a analizar
fr = np.random.uniform(-2, 2, realizaciones)  # vector de frecuencias aleatorias


# ==============================
# Ventanas a analizar
# ==============================
ventanas = {
    "Rectangular": np.ones(nn),
    "Flattop": get_window("flattop", nn),
    "Blackman-Harris": get_window("blackmanharris", nn),
    "Hamming": get_window("hamming", nn)
}


# ==============================
# Loop sobre SNRs
# ==============================
for snr in snrs:
    print(f"\n\n========== Resultados para SNR = {snr} dB ==========")

    # Cálculo de potencia de ruido para este SNR
    pot_ruido = a0**2 / (2*10**(snr/10))
    sigma_ruido = np.sqrt(pot_ruido)

    # ==============================
    # Construcción de realizaciones
    # ==============================
    tt = np.arange(nn).reshape(-1,1) / fs                 # eje de tiempo
    fr_mat = np.tile(fr, (nn, 1))                         # (nn x realizaciones)
    xx_mat = a0 * np.sin(2*np.pi*((nn/4)+fr_mat)*(fs/nn)*tt)  # matriz senoidales
    ruido_mat = np.random.normal(0, sigma_ruido, (nn, realizaciones))
    xx_mat_ruido = xx_mat + ruido_mat                     # señales con ruido

    # ==============================
    # Estimadores para cada ventana
    # ==============================
    resultados_amp = {}
    resultados_freq = {}

    for nombre, win in ventanas.items():
        # Aplicar ventana
        win = win.reshape(-1,1)
        xx_win = xx_mat_ruido * win

        # FFT normalizada
        X_win = (1/nn) * fft(xx_win, axis=0)

        # --------------------------
        # Estimador de amplitud a1
        # --------------------------
        gain = np.sum(win)/nn
        estim_amp = np.abs(X_win[nn//4,:]) / gain
        mu_amp = np.mean(estim_amp)
        sa = mu_amp - a0
        va = np.var(estim_amp)
        resultados_amp[nombre] = (sa, va)
        
        # --------------------------
        # Estimador de frecuencia Ω1
        # --------------------------
        espectro = np.abs(X_win)
        estim_freq_idx = np.argmax(espectro, axis=0)      # índices máximos
        estim_freq = estim_freq_idx * (fs/nn)             # en Hz

        freq_real = (nn/4 + fr) * (fs/nn)                   # frecuencias reales
        sa_f = np.mean(estim_freq - freq_real)
        va_f = np.var(estim_freq - freq_real)
        resultados_freq[nombre] = (sa_f, va_f)

    # ==============================
    # Mostrar resultados en tablas
    # ==============================
    print("\n--- Estimación de Amplitud ---")
    print("Ventana\t\tSesgo (sa)\tVarianza (va)")
    for nombre, (sa, va) in resultados_amp.items():
        print(f"{nombre:15s}\t{sa: .4e}\t{va: .4e}")

    print("\n--- Estimación de Frecuencia ---")
    print("Ventana\t\tSesgo (sa)\tVarianza (va)")
    for nombre, (sa, va) in resultados_freq.items():
        print(f"{nombre:15s}\t{sa: .4e}\t{va: .4e}")


# ==============================
# BONUS: comparar histogramas
# ==============================
plt.figure(figsize=(10,5))
for nombre, win in ventanas.items():
    win = win.reshape(-1,1)
    xx_win = xx_mat_ruido * win
    X_win = (1/nn) * fft(xx_win, axis=0)
    estim_amp = np.abs(X_win[nn//4,:])
    plt.hist(20*np.log10(estim_amp), bins=15, alpha=0.5, label=nombre)

plt.xlabel("Amplitud [dB]")
plt.ylabel("Cantidad de realizaciones")
plt.title(f"Comparación de ventanas en frecuencia N/4 ({fs/4:.2f} Hz)")
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# BONUS: Zero-padding y Estimadores Alternativos
# ==============================
resultados_freq_zp = {}
resultados_alt = {}

zp_factor = 4  # factor de zero-padding

for nombre, win in ventanas.items():
    win = win.reshape(-1,1)
    xx_win = xx_mat_ruido * win

    # -------- Zero-padding --------
    X_zp = (1/nn) * fft(xx_win, n=zp_factor*nn, axis=0)
    espectro_zp = np.abs(X_zp)
    estim_freq_idx_zp = np.argmax(espectro_zp, axis=0)
    estim_freq_zp = estim_freq_idx_zp * (fs/(zp_factor*nn))
    mu_freq_zp = np.mean(estim_freq_zp)
    freq_real = (nn/4 + fr) * (fs/nn)   # igual que antes
    sa_f_zp = np.mean(estim_freq_zp - freq_real)
    va_f_zp = np.var(estim_freq_zp - freq_real)
    resultados_freq_zp[nombre] = (sa_f_zp, va_f_zp)

    # -------- Estimador alternativo de amplitud --------
    vecindad = 2
    energia = np.sum(np.abs(X_win[(nn//4-vecindad):(nn//4+vecindad+1),:])**2, axis=0)
    gain = np.sum(win)/nn
    estim_amp_alt = np.sqrt(2*energia) / gain
    mu_amp_alt = np.mean(estim_amp_alt)
    sa_alt = mu_amp_alt - a0
    va_alt = np.var(estim_amp_alt)

    # -------- Estimador alternativo de frecuencia (interpolación cuadrática) --------
    X_win = (1/nn) * fft(xx_win, axis=0)
    espectro = np.abs(X_win)
    kmax = np.argmax(espectro, axis=0)
    alpha = espectro[(kmax-1)%nn, range(realizaciones)]
    beta  = espectro[kmax, range(realizaciones)]
    gamma = espectro[(kmax+1)%nn, range(realizaciones)]
    p = 0.5*(alpha - gamma)/(alpha - 2*beta + gamma + 1e-12)
    estim_freq_alt = (kmax + p) * (fs/nn)
    mu_freq_alt = np.mean(estim_freq_alt)
    sa_f_alt = np.mean(estim_freq_alt - freq_real)
    va_f_alt = np.var(estim_freq_alt - freq_real)

    resultados_alt[nombre] = ((sa_alt, va_alt), (sa_f_alt, va_f_alt))


# ==============================
# Mostrar resultados de Bonus
# ==============================
print("\n--- BONUS: Zero-Padding en Frecuencia ---")
print("Ventana\t\tSesgo (sa)\tVarianza (va)")
for nombre, (sa, va) in resultados_freq_zp.items():
    print(f"{nombre:15s}\t{sa: .4e}\t{va: .4e}")

print("\n--- BONUS: Estimadores Alternativos ---")
print("Ventana\t\tSesgo Amp\tVarianza Amp\tSesgo Freq\tVarianza Freq")
for nombre, ((sa_a, va_a), (sa_f, va_f)) in resultados_alt.items():
    print(f"{nombre:15s}\t{sa_a: .4e}\t{va_a: .4e}\t{sa_f: .4e}\t{va_f: .4e}")


