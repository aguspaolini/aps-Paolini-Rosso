# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 19:19:48 2025

@author: Admin
"""
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import get_window


def funcion_sen(ff, nn, amp=1, dc=0, ph=0, fs=2):
    ts = 1/fs  # período de muestreo [s]
    tt = np.arange(nn) * ts
    xx = amp*np.sin(2*np.pi*ff*tt + ph) + dc
    return tt, xx


fs = 1000  # Hz
nn = fs    # muestras
snr = 50   # dB
a0 = np.sqrt(2)  # Volts
realizaciones = 200
fr = np.random.uniform(-2, 2, realizaciones)  # vector de frecuencias random

# señal 1D de prueba
tt, s_1 = funcion_sen(ff=((fs/4)+fr[0])*(fs/nn), nn=nn, amp=a0, fs=fs)

# --- Cálculo de potencia de ruido ---
pot_ruido = a0**2 / (2*10**(snr/10))
print(f'Potencia teórica del ruido = {pot_ruido:3.3e}')

ruido = np.random.normal(0, np.sqrt(pot_ruido), nn)
var_ruido = np.var(ruido)
print(f'Potencia del ruido generado = {var_ruido:3.3e}')

# señal con ruido (1D, solo para comparar)
x_1 = s_1 + ruido
S_1 = (1/nn)*fft(s_1)
R = (1/nn)*fft(ruido)
X_1 = (1/nn)*fft(x_1)

ff_eje = np.arange(nn)*(fs/nn)

# plt.figure(figsize=(10, 4))
# plt.plot(ff_eje, 10*np.log10(2*np.abs(X_1)**2), '-', label="Señal con ruido")
# plt.xlim([0, fs/2])
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel('Amplitud [dB]')
# plt.title("FFT señal con ruido (una realización)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# ================================
# Ahora lo mismo pero en matriz
# ================================

tt = tt.reshape(-1, 1)        # reshape para que las filas sean tiempo
tt_mat = np.tile(tt, (1, realizaciones))     # (1000 x 200)

# Repetir fr en filas (mismo valor de frecuencia por columna)
fr_mat = np.tile(fr, (nn, 1))                # (1000 x 200)

# matriz de senoidales (sin ruido aún)
xx_mat = a0 * np.sin(2*np.pi*((nn/4)+fr_mat)*(fs/nn)*tt_mat)

# matriz de ruido: cada columna es una realización independiente
ruido_mat = np.random.normal(0, np.sqrt(pot_ruido), (nn, realizaciones))

# matriz de señales con ruido
xx_mat_ruido = xx_mat + ruido_mat

# Agrego ventaneo

flattop = get_window("flattop", nn)
flattop_mat = flattop.reshape(-1,1)      # shape (nn, 1) para broadcasting

b_harris = get_window("blackmanharris", nn)
b_harris_mat = b_harris.reshape(-1,1)

hamming = get_window("hamming", nn)
hamming_mat = hamming.reshape(-1,1)

rectang = np.ones(nn)
rectang_mat = rectang.reshape(-1, 1)

xx_vent_flat = xx_mat_ruido * flattop_mat
xx_vent_bharris = xx_mat_ruido * b_harris_mat
xx_vent_ham = xx_mat_ruido * hamming_mat
xx_vent_rect = xx_mat_ruido * rectang_mat

# FFT normalizada con ventana
X_mat_vent_flat = (1/nn) * fft(xx_vent_flat, axis=0)
X_mat_vent_bharris = (1/nn) * fft(xx_vent_bharris, axis=0)
X_mat_vent_ham = (1/nn) * fft(xx_vent_ham, axis=0)
X_mat_vent_rect = (1/nn) * fft(xx_vent_rect, axis=0)


# FFT normalizada a lo largo del eje del tiempo (filas)
X_mat = (1/nn) * fft(xx_mat_ruido, axis=0)

# Graficar todas las realizaciones
plt.figure(figsize=(10, 4))
plt.plot(ff_eje, 10*np.log10(2*np.abs(X_mat)**2), alpha=0.3)
plt.xlim([0, fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Amplitud [dB]')
plt.title("FFT de todas las realizaciones con ruido")
plt.grid(True)
plt.tight_layout()
plt.show()

# Eje de frecuencia correspondiente
ff_n4 = ff_eje[nn//4]

# Graficar todas las realizaciones flattop
plt.figure(figsize=(10, 4))
plt.plot(ff_eje, 10*np.log10(2*np.abs(X_mat_vent_flat)**2), alpha=0.3)
plt.xlim([0, fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Amplitud [dB]')
plt.title("FFT de todas las realizaciones con ruido (flattop)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ff_eje, 10*np.log10(2*np.abs(X_mat_vent_bharris)**2), alpha=0.3)
plt.xlim([0, fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Amplitud [dB]')
plt.title("FFT de todas las realizaciones con ruido (Blackman-Harris)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ff_eje, 10*np.log10(2*np.abs(X_mat_vent_rect)**2), alpha=0.3)
plt.xlim([0, fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Amplitud [dB]')
plt.title("FFT de todas las realizaciones con ruido (rectangular)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ff_eje, 10*np.log10(2*np.abs(X_mat_vent_ham)**2), alpha=0.3)
plt.xlim([0, fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Amplitud [dB]')
plt.title("FFT de todas las realizaciones con ruido (Hamming)")
plt.grid(True)
plt.tight_layout()
plt.show()

estimador_flat = np.abs(X_mat_vent_flat[nn//4,:])
# plt.hist(estimador_flat)
# plt.show()

estimador_rect = np.abs(X_mat_vent_rect[nn//4,:])
# plt.hist(estimador_rect)
# plt.show()

estimador_ham = np.abs(X_mat_vent_ham[nn//4,:])
# plt.hist(estimador_ham)
# plt.show()

estimador_bharris = np.abs(X_mat_vent_bharris[nn//4,:])
# plt.hist(estimador_bharris)
# plt.show()

plt.figure(figsize=(10,5))
plt.hist(10*np.log10(2*np.abs(estimador_rect)**2), bins=10, alpha=0.5, label='Rectangular')
plt.hist(10*np.log10(2*np.abs(estimador_ham)**2), bins=10, alpha=0.5, label='Hamming')
plt.hist(10*np.log10(2*np.abs(estimador_bharris)**2), bins=10, alpha=0.5, label='Blackman-Harris')
plt.hist(10*np.log10(2*np.abs(estimador_flat)**2), bins=10, alpha=0.5, label='Flattop')
plt.xlabel("Amplitud [dB]")
plt.ylabel("Cantidad de realizaciones")
plt.title(f"Comparación de ventanas en frecuencia N/4 ({ff_eje[nn//4]:.2f} Hz)")
plt.legend()
plt.grid(True)
plt.show()
