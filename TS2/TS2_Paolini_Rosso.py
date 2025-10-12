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
nn = 1200             # cantidad de muestras
tt = np.arange(nn) * ts  # Vector de tiempo

# ----------------------------- Sistema dado ------------------------------
# y[n] = 0.03 x[n] + 0.05 x[n-1] + 0.03 x[n-2] + 1.5 y[n-1] - 0.5 y[n-2]
b = np.array([0.03, 0.05, 0.03])    #Entrada x
a = np.array([1, -1.5, 0.5])        #Salida y (muevo hacia el otro lado)

# ----------------------------- Respuesta al impulso ----------------------
imp = np.zeros(nn)
imp[0] = 1.0
h_1 = lfilter(b, a, imp)

plt.figure(figsize=(8,4))
plt.plot(tt, h_1, 'x', color='peru', label='h_1[n]')
plt.title('Salida: Respuesta al impulso')
plt.legend(loc='center right')
plt.xlabel('Tiempo [s]')
plt.xlim(0,0.0003)
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.tight_layout()
plt.show()



#--------------------- Señales TS1 ------------------------------
# 1) Senoidal 2 kHz
ff_1 = 2000  # Hz
sen_1 = np.sin(2*np.pi*ff_1*tt)


def energia_discreta(x, ts):
    return np.sum(np.abs(x)**2) * ts

def potencia_discreta(x):
    return np.mean(np.abs(x)**2)


# -------- Simulación de una señal --------
def simular_senal(x, tt, nombre, b, a, h, fs, ts, graficar=True):
    
    # Respuesta usando lfilter
    y = lfilter(b, a, x)                    #Método por ecuación en diferencias
    
    # Respuesta con convolución directa
    y_conv = np.convolve(x, h)[:len(x)]     #Método con respuesta al impulso
    
    if graficar:
        plt.figure(figsize=(8,4))
        plt.plot(tt, y, 'o', color='aqua', label='Salida (lfilter)', markersize=3)
        plt.plot(tt, y_conv, 'x', color='deeppink', label='Salida (conv h_1)', markersize=1.5)
        plt.title(f'Salida para {nombre}')
        plt.legend(loc='center right')
        plt.xlabel('Tiempo [s]')
        plt.xlim(0,0.012)
        plt.ylabel('Amplitud [V]')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    print('\n')
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


simular_senal(sen_1, tt, 'Sinusoidal 2 kHz', b, a, h_1, fs, ts, graficar=True)

# 2) Amplificada y desfasada π/2
sen_2 = 4*np.sin(2*np.pi*ff_1*tt + np.pi/2)

simular_senal(sen_2, tt, 'Sinusoidal amplificada y desfasada', b, a, h_1, fs, ts, graficar=True)

# 3) Modulada en amplitud
ff_mod = 1000
m = 0.7
s_moduladora = np.sin(2*np.pi*ff_mod*tt)
sen_3 = (1 + m*s_moduladora) * sen_1

simular_senal(sen_3, tt, 'Modulada', b, a, h_1, fs, ts, graficar=True)

# 4) Señal recortada en amplitud al 75%
def recorte_por_amplitud(x, factor=0.75):
    A = np.max(np.abs(x))
    u = factor * A
    return np.clip(x, -u, u), u

sen_4, _ = recorte_por_amplitud(sen_1, 0.75)

simular_senal(sen_4, tt, 'Sinusoidal recortada', b, a, h_1, fs, ts, graficar=True)

# 5) Cuadrada 4 kHz
ff_2 = 4000
sen_5 = square(2*np.pi*ff_2*tt)

simular_senal(sen_5, tt, 'Cuadrada 4 kHz', b, a, h_1, fs, ts, graficar=True)

# 6) Pulso rectangular 10 ms dentro de ventana de 12 ms
nn_pulso = int(12e-3*fs)
tt_pulso = np.arange(nn_pulso)*ts
sen_6 = np.zeros_like(tt_pulso)
n_pulso = int(10e-3*fs)
sen_6[:n_pulso] = 1.0

simular_senal(sen_6, tt_pulso, 'Pulso 10 ms', b, a, h_1, fs, ts, graficar=True)


# ----------------------------- Ejercicio 2 -------------------------------

# Entrada sinusoidal de prueba
f_seno = 2000
x_seno = np.sin(2*np.pi*f_seno*tt)

# Sistema A: y[n] = x[n] + 3 x[n-10]
demora = 10
hA = np.zeros(nn)  #respuesta al impulso sistema A
hA[0] = 1
if demora < nn:
    hA[demora] = 3      #Hay salida 1 en n=0 y 3 en n=10  (necesito delta[0])

# Salida sistema A
yA = np.convolve(x_seno, hA)[:nn]  #y[n] = x[n] * hA[n].


# -------- Graficar respuesta al impulso --------
plt.figure(figsize=(8,4))
plt.plot(hA[:20], 'o', label='h_A[n]')
plt.title("Respuesta al impulso (primeros 20 valores)")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.xlim(0, 20)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Graficar salida --------
plt.figure(figsize=(8,4))
plt.plot(tt, x_seno, 'b.', label='Entrada x[n]', markersize=2)
plt.plot(tt, yA, 'o', color='m', label='Salida y[n]', markersize=3)
plt.xlim(0, 0.012)
plt.title("Comparación entrada vs salida sistema A")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


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

# -------- Graficar respuesta al impulso --------
plt.figure(figsize=(8,4))
plt.plot(hB[:50], 'o', label='h_B[n]')
plt.title("Respuesta al impulso (primeros 50 valores)")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.xlim(0,50)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Graficar salidas --------
plt.figure(figsize=(8,4))
plt.plot(tt, x_seno, 'b.', label='Entrada x[n]', markersize=2)
plt.plot(tt, yB, 'mo', label='Salida y[n]', markersize=3)
plt.title("Comparación entrada vs salida sistema B")
plt.xlabel("Tiempo [s]")
plt.xlim(0, 0.002)
plt.ylim(-2, 3)
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ----------------------------- Bonus: Windkessel -------------------------
# Parámetros fisiológicos típicos
C = 1.2   # Compliance [L/mmHg]
R = 1      # Resistencia [mmHg·s/ml]

fs_b = 1000          
ts_b = 1/fs_b          
nn_b = 2000         
tt_b = np.arange(nn_b) * ts_b   

# Flujo de entrada Q[n] 
f_q = 1.3
Q = 300 * np.maximum(0, np.sin(2*np.pi*f_q*tt_b))   # [ml/s]


P = np.zeros(nn_b)
P[0] = 80   # condición inicial [mmHg]
for n in range(nn_b-1):
    dP = (1/C) * (Q[n] - (1/R)*P[n])
    P[n+1] = P[n] + ts_b * dP

# Gráficos
fig, axs = plt.subplots(2, 1, figsize=(10,7))

# Presión arterial
axs[0].plot(tt_b, P, color="turquoise")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_xlim(0, 2)
axs[0].set_ylabel("Presión [mmHg]")
axs[0].set_title("Presión arterial P[n]")
axs[0].grid(True)

# Flujo sanguíneo
axs[1].plot(tt_b, Q, color="m")
axs[1].set_xlabel("Tiempo [s]")
axs[1].set_xlim(0, 2)
axs[1].set_ylabel("Flujo [ml/s]")
axs[1].set_title("Flujo de entrada Q[n]")
axs[1].grid(True)

plt.tight_layout()
plt.show()
