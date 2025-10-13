# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 15:45:20 2025

@author: Admin
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

# =======================================================
# CONFIGURACIÓN Y LECTURA DE SEÑALES
# =======================================================

# --- ECG sin ruido ---
fs_ecg = 1000  # Hz
ecg_sin_ruido = np.load('ecg_sin_ruido.npy').flatten()

# --- ECG con ruido (BONUS) ---
mat_struct = sio.loadmat('ECG_TP4.mat')
ecg_con_ruido = mat_struct['ecg_lead'].flatten()

# --- PPG sin ruido ---
fs_ppg = 400  # Hz
ppg = np.load('ppg_sin_ruido.npy').flatten()

# --- Audio ---
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
wav_data = wav_data.astype(float)

# =======================================================
# FUNCIÓN GENERAL DE PSD Y ANCHO DE BANDA
# =======================================================
def calcular_psd_y_bw(signal, fs, n_promedios, nfft_factor, color, nombre, fmax=None, mostrar_min=False):
    """
    Calcula PSD mediante Welch con ventana Hamming.
    Estima el ancho de banda al 99% de la potencia acumulada.
    Si mostrar_min=True, también marca el 1% (ancho de banda mínimo).
    Retorna el ancho o rango de banda (Hz).
    """
    nperseg = signal.shape[0] // n_promedios
    nfft = nfft_factor * nperseg

    f, Pxx = sig.welch(signal, fs=fs, window='hamming', nperseg=nperseg, nfft=nfft)
    
    df = f[1] - f[0]
    pot_total = np.sum(Pxx) * df
    pot_acumulada = np.cumsum(Pxx) * df

    # índices del 1% y 99% de energía acumulada
    indice_99 = np.where(pot_acumulada >= 0.99 * pot_total)[0][0]
    bw_max = f[indice_99]

    if mostrar_min:
        indice_01 = np.where(pot_acumulada >= 0.01 * pot_total)[0][0]
        bw_min = f[indice_01]
    else:
        bw_min = None

    # Gráfico PSD
    plt.figure()
    plt.plot(f, Pxx, color=color)
    if mostrar_min:
        plt.axvline(bw_min, color=color, linestyle='--', alpha=0.6)
        plt.axvline(bw_max, color=color, linestyle='--', alpha=0.6,
                    label=f'BW = {bw_max - bw_min:.1f} Hz')
    else:
        plt.axvline(bw_max, color=color, linestyle='--',
                    label=f'BW = {bw_max:.2f} Hz')

    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Densidad espectral [V²/Hz]')
    plt.title(f'Densidad espectral de potencia - {nombre}')
    plt.grid(True)
    if fmax:
        plt.xlim(0, fmax)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Devuelve ancho de banda (rango si corresponde)
    if mostrar_min:
        return bw_min, bw_max, bw_max - bw_min
    else:
        return bw_max

# =======================================================
# CÁLCULO DE PSD Y ANCHOS DE BANDA
# =======================================================

# --- ECG sin ruido ---
bw_ecg_sin = calcular_psd_y_bw(ecg_sin_ruido, fs_ecg, n_promedios=30, nfft_factor=5,
                               color='r', nombre='ECG (sin ruido)', fmax=50)

# --- PPG sin ruido ---
bw_ppg = calcular_psd_y_bw(ppg, fs_ppg, n_promedios=45, nfft_factor=2,
                           color='g', nombre='PPG (sin ruido)', fmax=20)

# --- Audio (con ancho mínimo y máximo) ---
bw_audio_min, bw_audio_max, bw_audio_total = calcular_psd_y_bw(
    wav_data, fs_audio, n_promedios=135, nfft_factor=4,
    color='m', nombre='Audio ("La Cucaracha")', fmax=4000, mostrar_min=True
)

# --- BONUS: ECG con ruido ---
bw_ecg_con = calcular_psd_y_bw(ecg_con_ruido, fs_ecg, n_promedios=30, nfft_factor=5,
                               color='orange', nombre='ECG (con ruido)', fmax=50)

# =======================================================
# TABLA FINAL DE RESULTADOS
# =======================================================
print("\n========== RESULTADOS FINALES ==========")
print(f"{'Señal':<30}{'Ancho de Banda [Hz]':>25}")
print("------------------------------------------------------------")
print(f"{'ECG (sin ruido)':<30}{bw_ecg_sin:>20.2f}")
print(f"{'PPG (sin ruido)':<30}{bw_ppg:>20.2f}")
print(f"{'Audio (voz)':<30}{bw_audio_total:>20.2f} (de {bw_audio_min:.1f} a {bw_audio_max:.1f})")
print(f"{'ECG (con ruido) [BONUS]':<30}{bw_ecg_con:>20.2f}")
print("------------------------------------------------------------")
