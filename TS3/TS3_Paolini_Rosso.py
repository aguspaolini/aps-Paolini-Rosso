# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 11:09:43 2025

@author: Admin
"""

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import freqz

def funcion_sen(ff, nn, amp=1, dc=0, ph=0, fs=2):
    ts = 1 / fs
    tt = np.arange(nn) * ts
    xx = amp * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx


fs = 1000             # frecuencia de muestreo [1/s]
N = fs                # cantidad de muestras = 1000
delta_f = fs / N      # resolución en frecuencia

#Frecuencias pedidas por la consigna
f1 = N / 4                    # k0 = N/4 (f0 exacta en un bin)
f2 = (N / 4 + 0.25) * delta_f #desintonía 0.25 Δf
f3 = (N / 4 + 0.5) * delta_f  #desintonía 0.5 Δf

# ----------- Señales en tiempo -----------
#amplitud = sqrt(2) para que la potencia sea unitaria
tt, x1 = funcion_sen(f1, N, amp=np.sqrt(2), fs=fs)
tt, x2 = funcion_sen(f2, N, amp=np.sqrt(2), fs=fs)
tt, x3 = funcion_sen(f3, N, amp=np.sqrt(2), fs=fs)


# ----------- FFT y PSD -----------
X1 = fft(x1)
PSD1 = (1 / N**2) * np.abs(X1) ** 2

X2 = fft(x2)
PSD2 = (1 / N**2) * np.abs(X2) ** 2

X3 = fft(x3)
PSD3 = (1 / N**2) * np.abs(X3) ** 2

ff_eje = np.arange(N) * delta_f

# ----------- Gráfico parte (a) -----------
plt.figure(figsize=(10, 4))
plt.plot(ff_eje, 10 * np.log10(PSD1), 's', label="f1 = N/4")
plt.plot(ff_eje, 10 * np.log10(PSD2), 'x', label="f2 = (N/4 + 0.25) Δf")
plt.plot(ff_eje, 10 * np.log10(PSD3), '.', label="f3 = (N/4 + 0.5) Δf")

plt.xlim([230, 270])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.title("Parte (a) - Efecto de desintonía")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------Parte (b): Parseval -----------

# Señal 1
energia_frec_1 = (1/N)*np.sum(np.abs(X1)**2)           #según Parseval, en frecuencia
energia_tiempo_1 = np.sum(np.abs(x1)**2)           #energía en tiempo


# Potencia unitaria
pot_frec_1 = energia_frec_1 /N       #dividir entre N para obtener potencia promedio
pot_tiempo_1 = energia_tiempo_1 /N
pot_PSD_1 = np.sum(PSD1)

print("Señal 1:")
print(f"Potencia tiempo = {pot_tiempo_1:.3f}")
print(f"Potencia frecuencia = {pot_frec_1:.3f}")
print(f"Potencia PSD = {pot_PSD_1:.3f}")
print(f"Energía tiempo = {energia_tiempo_1:.3f}")
print(f"Energía frecuencia = {energia_frec_1:.3f}")
if np.isclose(energia_frec_1, energia_tiempo_1):
    print("Se cumple Parseval.")
if np.isclose(pot_PSD_1, pot_tiempo_1):
    print("✓ PSD consistente con Parseval")
print(" ")


# Señal 2
energia_frec_2 = (1/N)*np.sum(np.abs(X2)**2)           
energia_tiempo_2 = np.sum(np.abs(x2)**2)           

# Potencia unitaria
pot_frec_2 = energia_frec_2 /N       
pot_tiempo_2 = energia_tiempo_2 /N
pot_PSD_2 = np.sum(PSD2)

print("Señal 2:")
print(f"Potencia tiempo = {pot_tiempo_2:.3f}")
print(f"Potencia frecuencia = {pot_frec_2:.3f}")
print(f"Potencia PSD = {pot_PSD_2:.3f}")
print(f"Energía tiempo = {energia_tiempo_2:.3f}")
print(f"Energía frecuencia = {energia_frec_2:.3f}")
if np.isclose(energia_frec_2, energia_tiempo_2):
    print("Se cumple Parseval.")
if np.isclose(pot_PSD_2, pot_tiempo_2):
    print("✓ PSD consistente con Parseval")
print(" ")

# Señal 3
energia_frec_3 = (1/N)*np.sum(np.abs(X3)**2)           
energia_tiempo_3 = np.sum(np.abs(x3)**2)          

pot_frec_3 = energia_frec_3 /N       
pot_tiempo_3 = energia_tiempo_3 /N
pot_PSD_3 = np.sum(PSD3)

print("Señal 3:")
print(f"Potencia tiempo = {pot_tiempo_3:.3f}")
print(f"Potencia frecuencia = {pot_frec_3:.3f}")
print(f"Potencia PSD = {pot_PSD_3:.3f}")
print(f"Energía tiempo = {energia_tiempo_3:.3f}")
print(f"Energía frecuencia = {energia_frec_3:.3f}")
if np.isclose(energia_frec_3, energia_tiempo_3):
    print("Se cumple Parseval.")
if np.isclose(pot_PSD_3, pot_tiempo_3):
    print("✓ PSD consistente con Parseval")
print(" ")


# ----------- Parte (c): Zero padding -----------
Npad = 10*N   #señal extendida con 9N ceros

def zero_padding_fft(x, Npad, fs):
    x_pad = np.zeros(Npad)
    x_pad[:len(x)] = x
    X_pad = fft(x_pad)
    PSD_pad = (1/(len(x)*len(x)))*np.abs(X_pad)**2   #normalizo por N original
    ff_pad = np.arange(Npad)*fs/Npad
    return ff_pad, PSD_pad

ff1_pad, PSD1_pad = zero_padding_fft(x1, Npad, fs)
ff2_pad, PSD2_pad = zero_padding_fft(x2, Npad, fs)
ff3_pad, PSD3_pad = zero_padding_fft(x3, Npad, fs)

plt.figure(figsize=(10,4))
plt.plot(ff1_pad, 10*np.log10(PSD1_pad), label="f1 = N/4 Δf (ZP)")
plt.plot(ff_eje, 10*np.log10(PSD1), 'x', label="f1 = N/4 Δf")
plt.xlim([230, 270])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.title("Parte (c) - Zero Padding - f1 = N/4 Δf")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(ff2_pad, 10*np.log10(PSD2_pad), label="f2 = (N/4 + 0.25) Δf (ZP)")
plt.plot(ff_eje, 10*np.log10(PSD2), 'x', label="f2 = (N/4 + 0.25) Δf")
plt.xlim([230, 270])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.title("Parte (c) - Zero Padding - f2 = (N/4 + 0.25) Δf")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,4))
plt.plot(ff3_pad, 10*np.log10(PSD3_pad), label="f3 = (N/4 + 0.5) Δf (ZP)")
plt.plot(ff_eje, 10*np.log10(PSD3), 'x', label="f3 = (N/4 + 0.5) Δf")
plt.xlim([230, 270])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.title("Parte (c) - Zero Padding - f3 = (N/4 + 0.5) Δf")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------- Bonus ----------------------------
#Respuesta en frecuencia

def plot_frec_respuesta(b, a, fs, titulo='Sistema LTI'):
    w, h = freqz(b, a, worN=2048)  #2048 puntos para buena resolución
    f = w * fs / (2*np.pi)          #Convertir de rad/muestra a Hz
    
    plt.figure(figsize=(10,5))
    
    # Magnitud
    plt.subplot(2,1,1)
    plt.plot(f, 20*np.log10(np.abs(h)), 'b')
    plt.title(f'Respuesta en frecuencia: {titulo}')
    plt.ylabel('Magnitud [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.xlim(0,500)
    plt.grid(True)
    
    # Fase
    plt.subplot(2,1,2)
    plt.plot(f, np.angle(h, deg=True), 'r')
    plt.xlabel('Frecuencia [Hz]')
    plt.xlim(0,500)
    plt.ylabel('Fase [°]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# 1) Sistema original (b, a)

b = np.array([0.03, 0.05, 0.03])    #Entrada x
a = np.array([1, -1.5, 0.5])        #Salida y (muevo hacia el otro lado)

plot_frec_respuesta(b, a, fs, 'Sistema original')

# 2) Sistema A: y[n] = x[n] + 3 x[n-10]
# H_A(z) = 1 + 3 z^-10
hA = np.zeros(N)  #respuesta al impulso sistema A
hA[0] = 1
if 10 < N:
    hA[10] = 3      #Hay salida 1 en n=0 y 3 en n=10  (necesito delta[0])
    
bA = hA[:10+1]   # coeficientes no nulos de hA
aA = [1]
plot_frec_respuesta(bA, aA, fs, 'Sistema A')

# 3) Sistema B: y[n] = x[n] + 3 y[n-10]
# bB y aB ya definidos
bB = [1]
aB = np.zeros(10+1)
aB[0] = 1
aB[10] = -3

plot_frec_respuesta(bB, aB, fs, 'Sistema B')

