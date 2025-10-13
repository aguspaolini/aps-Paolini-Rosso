# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 15:35:24 2025

@author: Admin
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

# =======================================================
# LECTURA DE SEÑALES Y CONFIGURACIÓN
# =======================================================

# =====================
# ECG (sin ruido)
# =====================
fs_ecg = 1000  # Hz
ecg_one_lead = np.load('ecg_sin_ruido.npy')

# =====================
# PPG (con ruido)
# =====================
fs_ppg = 400  # Hz
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)
ppg = ppg.flatten()

# =====================
# Audio (con ruido)
# =====================
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
wav_data = wav_data.astype(float)

# =======================================================
# FUNCIÓN GENERAL DE PSD Y ANCHO DE BANDA
# =======================================================
def calcular_psd_y_bw(signal, fs, n_promedios, nfft_factor, color, nombre, fmax=None):
    """
    Calcula PSD por Welch, estima ancho de banda al 99% y grafica resultado.
    Retorna el ancho de banda (Hz).
    """
    nperseg = signal.shape[0] // n_promedios
    nfft = nfft_factor * nperseg

    f, Pxx = sig.welch(signal, fs=fs, window='hamming', nperseg=nperseg, nfft=nfft)
    
    df = f[1] - f[0]
    pot_total = np.sum(Pxx) * df
    pot_acumulada = np.cumsum(Pxx) * df

    indice_99 = np.where(pot_acumulada >= 0.99 * pot_total)[0][0]
    bw = f[indice_99]

    # Gráfico de PSD
    plt.figure()
    plt.plot(f, Pxx, color=color)
    plt.axvline(bw, color=color, linestyle='--', label=f'BW = {bw:.2f} Hz')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Densidad espectral [V²/Hz]')
    plt.title(f'Densidad espectral de potencia - {nombre}')
    plt.grid(True)
    if fmax:
        plt.xlim(0, fmax)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return bw

# =======================================================
# CÁLCULOS DE PSD Y ANCHO DE BANDA
# =======================================================

# --- ECG ---
promedios_ecg = 30
bw_ecg = calcular_psd_y_bw(ecg_one_lead, fs_ecg, promedios_ecg, nfft_factor=5, color='r', nombre='ECG', fmax=50)
print(f"Ancho de banda ECG: {bw_ecg:.2f} Hz")

# --- PPG ---
promedios_ppg = 45
bw_ppg = calcular_psd_y_bw(ppg, fs_ppg, promedios_ppg, nfft_factor=2, color='g', nombre='PPG', fmax=20)
print(f"Ancho de banda PPG: {bw_ppg:.2f} Hz")

# --- Audio ---
promedios_audio = 135
nperseg_audio = wav_data.shape[0] // promedios_audio
nfft_audio = 4 * nperseg_audio

f3, Pxx3 = sig.welch(wav_data, fs=fs_audio, window='hamming', nperseg=nperseg_audio, nfft=nfft_audio)
df3 = f3[1] - f3[0]
pot_total3 = np.sum(Pxx3) * df3
acum3 = np.cumsum(Pxx3) * df3

idx_min = np.where(acum3 >= 0.01 * pot_total3)[0][0]
idx_max = np.where(acum3 >= 0.99 * pot_total3)[0][0]
bw_audio_min = f3[idx_min]
bw_audio_max = f3[idx_max]
bw_audio = bw_audio_max - bw_audio_min

# --- Gráfico de Audio ---
plt.figure()
plt.plot(f3, Pxx3, color='m')
plt.axvline(bw_audio_min, color='m', linestyle='--')
plt.axvline(bw_audio_max, color='m', linestyle='--', label=f'BW = {bw_audio:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral [V²/Hz]')
plt.title('Densidad espectral de potencia - Audio ("La Cucaracha")')
plt.xlim(500, 3000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Ancho de banda Audio: {bw_audio_min:.1f} Hz – {bw_audio_max:.1f} Hz (Δf = {bw_audio:.1f} Hz)")


# =======================================================
# TABLA FINAL DE RESULTADOS
# =======================================================
print("\n========== RESULTADOS FINALES ==========")
print(f"{'Señal':<10}{'Ancho de Banda [Hz]':>25}")
print("-----------------------------------------")
print(f"{'ECG':<10}{bw_ecg:>20.2f}")
print(f"{'PPG':<10}{bw_ppg:>20.2f}")
print(f"{'Audio':<10}{bw_audio:>20.2f}")
print("-----------------------------------------")

