# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 21:45:47 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, square


fs = 100000        # frecuencia de muestreo [Hz]
ts = 1 / fs
nn = 1500              # cantidad de muestras
tt = np.arange(nn) * ts  # Vector de tiempo

# ----------------------------- Sistema dado ------------------------------
# y[n] = 0.03 x[n] + 0.05 x[n-1] + 0.03 x[n-2] + 1.5 y[n-1] - 0.5 y[n-2]
b = np.array([0.03, 0.05, 0.03])    #Entrada x
a = np.array([1, -1.5, 0.5])        #Salida y (muevo hacia el otro lado)

#--------------------- Señales TS1 ------------------------------
# 1) Senoidal 2 kHz
ff_1 = 2000  # Hz
sen_1 = np.sin(2*np.pi*ff_1*tt)

# 2) Amplificada y desfasada π/2
sen_2 = 4*np.sin(2*np.pi*ff_1*tt + np.pi/2)

# 3) Modulada en amplitud (con 1 kHz)
ff_mod = 1000
m = 0.7
s_moduladora = np.sin(2*np.pi*ff_mod*tt)
sen_3 = (1 + m*s_moduladora) * sen_1

# 4) Señal recortada en amplitud al 75%
def recorte_por_amplitud(x, factor=0.75):
    A = np.max(np.abs(x))
    u = factor * A
    return np.clip(x, -u, u), u

sen_4, _ = recorte_por_amplitud(sen_1, 0.75)

# 5) Cuadrada 4 kHz
ff_2 = 4000
sen_5 = square(2*np.pi*ff_2*tt)

# 6) Pulso rectangular 10 ms dentro de ventana de 15 ms
nn_pulso = int(15e-3*fs)
tt_pulso = np.arange(nn_pulso)*ts
sen_6 = np.zeros_like(tt_pulso)
n_pulso = int(10e-3*fs)
sen_6[:n_pulso] = 1.0


# ----------------------------- Respuesta al impulso ----------------------
imp = np.zeros(nn)
imp[0] = 1.0
h = lfilter(b, a, imp)

plt.figure(figsize=(10,6))
plt.plot(tt, h, '--', label='h')
plt.title('Salida impulso')
plt.legend()
plt.xlabel('Tiempo [s]')
plt.xlim(0,0.015)
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nSistema 1: h (primeros 20 valores):", h[:20])

def energia_discreta(x, ts):
    return np.sum(np.abs(x)**2) * ts

def potencia_discreta(x):
    return np.mean(np.abs(x)**2)


# -------- Simulación de una señal --------
def simular_senal(x, tt, nombre, b, a, h, fs, ts, graficar=True):
    
    # Respuesta usando lfilter
    y = lfilter(b, a, x)
    
    # Respuesta con convolución directa
    y_conv = np.convolve(x, h)[:len(x)]
    
    # Graficar
    if graficar:
        plt.figure(figsize=(10,6))
        plt.plot(tt, y, label='Salida (lfilter)')
        plt.plot(tt, y_conv, '--', label='Salida (conv h)')
        plt.title(f'Salida para {nombre}')
        plt.legend()
        plt.xlabel('Tiempo [s]')
        plt.xlim(0,0.012)
        plt.ylabel('Amplitud [V]')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Mostrar resultados
    print('---')
    print(f'Entrada: {nombre}')
    print(f'Frecuencia de muestreo: {fs} Hz')
    print(f'Tiempo de simulación: {len(x)/fs:.4f} s')
    
    
    # Energía/potencia
    if nombre == 'Pulso 10 ms':
        energia_lfilter = energia_discreta(y, ts)
        energia_conv = energia_discreta(y_conv, ts)
        print(f'Energía de la salida con lfilter: {energia_lfilter:.4f} J')
        print(f'Energía de la salida con respuesta al impulso: {energia_conv:.4f} J')
        
    else:
        potencia_lfilter = potencia_discreta(y)
        potencia_conv = potencia_discreta(y_conv)
        print(f'Potencia de la salida con lfilter: {potencia_lfilter:.4f} W')
        print(f'Potencia de la salida con respuesta al impulso: {potencia_conv:.4f} W')


simular_senal(sen_1, tt, 'Sinusoidal 2 kHz', b, a, h, fs, ts, graficar=True)
simular_senal(sen_2, tt, 'Sinusoidal amplificada y desfasada', b, a, h, fs, ts, graficar=True)
simular_senal(sen_3, tt, 'Modulada', b, a, h, fs, ts, graficar=True)
simular_senal(sen_4, tt, 'Sinusoidal recortada', b, a, h, fs, ts, graficar=True)
simular_senal(sen_5, tt, 'Cuadrada 4 kHz', b, a, h, fs, ts, graficar=True)
simular_senal(sen_6, tt_pulso, 'Pulso 10 ms', b, a, h, fs, ts, graficar=True)


# ----------------------------- Ejercicio 2 -------------------------------

# Entrada senoidal de prueba
f_seno = 2000
x_seno = np.sin(2*np.pi*f_seno*tt)


demora = 10
# Sistema A: y[n] = x[n] + 3 x[n-10]
hA = np.zeros(nn)  #respuesta al impulso sistema A
hA[0] = 1
if demora < nn:
    hA[demora] = 3      #Hay salida 1 en n=0 y 3 en n=10  (necesito delta[0])

# Salida sistema A
yA = np.convolve(x_seno, hA)[:nn]  #y[n] = x[n] * hA[n].


# Sistema B: y[n] = x[n] + 3 y[n-10]
bB = [1]
aB = np.zeros(demora+1)
aB[0] = 1
aB[demora] = -3
imp = np.zeros(nn)
imp[0] = 1                   # pulso unitario
hB = lfilter(bB, aB, imp)    # respuesta al impulso usando lfilter


# Salida sistema B
yB = lfilter(bB, aB, x_seno)

# -------- Graficar respuestas al impulso --------
plt.figure(figsize=(10,4))
plt.plot(hA[:50], 'o', label='h_A[n]')
plt.plot(hB[:50], 'x', label='h_B[n]')
plt.title("Respuestas al impulso (primeros 50 valores)")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Graficar salidas --------
plt.figure(figsize=(10,5))
plt.plot(tt, yA, label='Salida sistema A')
plt.plot(tt, yB, '--', label='Salida sistema B')
plt.xlim(0, 0.012)
plt.title("Salida de ambos sistemas ante sinusoidal de 2 kHz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nSistema A: hA (primeros 20 valores):", hA[:20])
print("Sistema B: hB (primeros 50 valores):", hB[:50])

# ----------------------------- Bonus: Windkessel -------------------------
C = 1.5e-3
R = 1.0

f_hr = 1.2
Q0 = 5e-3
Q = Q0 + 2e-3 * np.maximum(0, np.sin(2*np.pi*f_hr*tt))

P = np.zeros(nn)
for n in range(nn-1):
    P[n+1] = P[n] + (ts/C) * ( Q[n] - (1.0/R) * P[n] )

plt.figure(figsize=(8,4))
plt.plot(tt, Q, label='Q (flujo)')
plt.plot(tt, P, label='P (presión)')
plt.title('Modelo Windkessel (Euler hacia adelante)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nWindkessel simulado con C=",C," R=",R," fs=",fs)
