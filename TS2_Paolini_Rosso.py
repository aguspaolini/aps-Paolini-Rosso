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
nn = 500              # cantidad de muestras
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

sen_4, _ = recorte_por_amplitud(sen_3, 0.75)

# 5) Cuadrada 4 kHz
ff_2 = 4000
sen_5 = square(2*np.pi*ff_2*tt)

# 6) Pulso rectangular 10 ms dentro de ventana de 15 ms
nn_pulso = int(15e-3*fs)
tt_pulso = np.arange(nn_pulso)*ts
sen_6 = np.zeros_like(tt_pulso)
n_pulso = int(10e-3*fs)
sen_6[:n_pulso] = 1.0

# Guardo todas
signals = {
    'senoidal_2kHz': (tt, sen_1),
    'senoidal_amp_desfase': (tt, sen_2),
    'senoidal_mod_AM': (tt, sen_3),
    'senoidal_recortada': (tt, sen_4),
    'cuadrada_4kHz': (tt, sen_5),
    'pulso_10ms': (tt_pulso, sen_6)
}

# ----------------------------- Respuesta al impulso ----------------------
imp = np.zeros(nn)
imp[0] = 1.0
h = lfilter(b, a, imp)


def energia_discreta(x, Ts):
    return np.sum(np.abs(x)**2) * Ts


# -------- Simulación de una señal --------
def simular_senal(x, tt, nombre, b, a, h, fs, ts, graficar=True):
    
    # Respuesta usando lfilter
    y = lfilter(b, a, x)
    
    # Respuesta con convolución directa
    y_conv = np.convolve(x, h)[:len(x)]
    
    # Energía/potencia
    energia = energia_discreta(y, ts)
    
    # Graficar
    if graficar:
        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(tt, x)
        plt.title(f'Entrada: {nombre}')
        plt.grid(True)
        
        plt.subplot(2,1,2)
        plt.plot(tt, y, label='Salida (lfilter)')
        plt.plot(tt, y_conv, '--', label='Salida (conv h)')
        plt.title(f'Salida para {nombre}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Mostrar resultados
    print('---')
    print(f'Entrada: {nombre}')
    print(f'Frecuencia de muestreo: {fs} Hz')
    print(f'Tiempo de simulación: {len(x)/fs:.6f} s')
    print(f'Energía de la salida: {energia:.6e}')
  
    return y, energia

y1, E1 = simular_senal(sen_1, tt, 'senoidal_2kHz', b, a, h, fs, ts, graficar=True)
y2, E2 = simular_senal(sen_2, tt, 'senoidal_amp_desfase', b, a, h, fs, ts, graficar=True)
y3, E3 = simular_senal(sen_3, tt, 'senoidal_mod_AM', b, a, h, fs, ts, graficar=True)
y4, E4 = simular_senal(sen_4, tt, 'senoidal_recortada', b, a, h, fs, ts, graficar=True)
y5, E5 = simular_senal(sen_5, tt, 'cuadrada_4kHz', b, a, h, fs, ts, graficar=True)
y6, E6 = simular_senal(sen_6, tt_pulso, 'pulso_10ms', b, a, h, fs, ts, graficar=True)


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

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(tt, x_seno)
plt.grid(True)
plt.title('Entrada seno 2kHz (para sistemas A y B)')

plt.subplot(2,1,2)
plt.plot(tt, yA, label='Salida A')
plt.plot(tt, yB, '--', label='Salida B')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nSistema A: hA (primeros 20 valores):", hA[:20])
print("Sistema B: hB (primeros 20 valores):", hB[:20])

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
