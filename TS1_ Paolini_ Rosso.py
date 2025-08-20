# -*- coding: utf-8 -*-
"""
Created on Fri Aug  14 17:05:23 2025

@author: aguspaolini
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
from scipy.io import wavfile

#%%                                  Punto 1

fs = 100000  # Frecuencia de muestreo [Hz]
ts = 1/fs     # Tiempo entre muestras [s]
nn = 500      # Cantidad de muestras → duración total 5 ms
tt = np.arange(nn)*ts   # Vector de tiempo

# Función para graficar y mostrar Ts, N y energía
def grafico(x, y, titulo):
    E = np.sum(np.abs(y)**2) * ts  # Energía de la señal
    N = len(y)                     # Cantidad de muestras
    Ts = x[1]-x[0]                 # Tiempo entre muestras
    
    plt.figure(figsize=(8,4))
    line_hdls = plt.plot(x, y, 'g', label='Señal')
    plt.title(titulo)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.grid(True)
    
    # Información en la leyenda
    info_text = f"Ts = {Ts*1e6:.2f} µs, N = {N}, Energía = {E:.4f}"
    plt.legend(line_hdls, [info_text], loc='upper right')
    
    plt.tight_layout()
    plt.show()


# Señal 1: senoidal 2 kHz
ff_1 = 2000  # Hz
sen_1 = np.sin(2*np.pi*ff_1*tt)
grafico(tt, sen_1, 'Señal senoidal 2 kHz (S1)')


# Señal 2: amplificada y desfasada
sen_2 = 4*np.sin(2*np.pi*ff_1*tt + np.pi/2)
grafico(tt, sen_2, 'S1 amplificada y desfasada π/2')


# Señal 3: modulada en amplitud
ff_mod = 1000  # Hz
s_moduladora = np.sin(2*np.pi*ff_mod*tt)
sen_3 = sen_1 * s_moduladora
plt.figure(figsize=(8,4))
plt.plot(tt, sen_3, 'g', label='S1 modulada en amplitud')
plt.plot(tt, s_moduladora, 'b--', label='Señal moduladora')
plt.legend()
plt.title('Modulación')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.tight_layout()
plt.show()


# Señal 4: al 75% de la energía de Señal 3
e_sen_3 = np.sum(np.abs(sen_3)**2) * ts  # Energía de la señal 3
e_objetivo = 0.75 * e_sen_3              # Energía buscada de la señal 4
sen_4 = np.sqrt(e_objetivo/e_sen_3) * sen_3
grafico(tt, sen_4, 'S1 modulada y recortada al 75% de energía')


# Señal 5: cuadrada 4 kHz
ff_2 = 4000  # Hz
sen_5 = signal.square(2*np.pi*ff_2*tt)
grafico(tt, sen_5, 'Señal cuadrada 4 kHz')


# Señal 6: pulso rectangular 10 ms

nn_pulso = int(15e-3*fs)   # número total de muestras dado 15 ms
tt_pulso = np.arange(nn_pulso)*ts   # vector de tiempo

sen_6 = np.zeros_like(tt_pulso)    # creo vector de igual longitud
n_pulso = int(10e-3*fs)            # duración del pulso = 10 ms
sen_6[:n_pulso] = 1                # vale 1 en esos 10 ms

grafico(tt_pulso, sen_6, 'Pulso rectangular 10 ms')


#%%                           Verifico ortogonalidad


def prod_interno(s1, s2, ts, n1, n2,  tol=1e-10):
  
    # Producto interno discreto
    p_int = np.sum(s1 * s2) * ts
    
    if np.abs(p_int) < tol:
        print(f'Las señales {n1} y {n2} son ortogonales.')
    else:
        print(f'Las señales {n1} y {n2} NO son ortogonales.')

# Verificar ortogonalidad
prod_interno(sen_1, sen_2, ts, 'S1', 'S2')
prod_interno(sen_1, sen_3, ts, 'S1', 'S3')
prod_interno(sen_1, sen_4, ts, 'S1', 'S4')
prod_interno(sen_1, sen_5, ts, 'S1', 'S5')
prod_interno(sen_1, sen_6[:len(sen_1)], ts, 'S1', 'Pulso')

#%%                       Punto 3

def correlacion(x, y, titulo):
    corr = np.correlate(x, y, mode='full') * ts
    lags = np.arange(-len(x)+1, len(x)) * ts
    
    plt.figure(figsize=(8,4))
    plt.plot(lags, corr, 'b')
    plt.title(titulo)
    plt.xlabel('Retraso [s]')
    plt.ylabel('Correlación')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Autocorrelación de S1
correlacion(sen_1, sen_1, 'Autocorrelación de S1')

# Correlación entre S1 y las demás
correlacion(sen_1, sen_2, 'Correlación S1-S2')
correlacion(sen_1, sen_3, 'Correlación S1-S3')
correlacion(sen_1, sen_4, 'Correlación S1-S4')
correlacion(sen_1, sen_5, 'Correlación S1-S5')
correlacion(sen_1, sen_6[:len(sen_1)], 'Correlación S1-Pulso')

#%%                        Punto 4

# Demostración:
# 2 sin(α) sin(β) = cos(α-β) - cos(α+β)

w = 2*np.pi*1000   # w = 2πf
alpha = w*tt       # α = ω·t
beta = (w/2)*tt    # β = ω/2·t   (la mitad de α)

lhs = 2*np.sin(alpha)*np.sin(beta)          # lado izquierdo
rhs = np.cos(alpha-beta) - np.cos(alpha+beta)  # lado derecho

plt.figure(figsize=(8,4))
plt.plot(tt, lhs, 'r', label='2 sin(α)·sin(β)')
plt.plot(tt, rhs, 'b--', label='cos(α-β) - cos(α+β)')
plt.legend()
plt.title('Verificación de la identidad trigonométrica')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.show()

#%%                         Bonus 4

# Parámetros
tiempo_total = 20    # duración en segundos
dt = 0.5             # distancia entre muestras (segundos)

tiempos = []
temps = []

plt.ion()
fig, ax = plt.subplots(figsize=(8,4))
t0 = time.time()

for _ in range(int(tiempo_total/dt)):
    ahora = time.time() - t0   # tiempo relativo desde inicio
    
    # SIMULACIÓN: temperatura alrededor de 50°C con fluctuaciones
    temp = 50 + 2*np.sin(0.2*ahora) + np.random.normal(0,0.3)
    
    tiempos.append(ahora)
    temps.append(temp)
    
    # Graficar en vivo
    ax.clear()
    ax.plot(tiempos, temps, '-o', label="Temperatura CPU (simulada)")
    ax.set_title("Temperatura CPU en tiempo real")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Temperatura [°C]")
    ax.grid(True)
    ax.legend()
    plt.pause(0.01)
    
    time.sleep(dt)

plt.ioff()
plt.show()


#%% ==================== Bonus 5: Procesar un WAV ====================

# Leer archivo wav
fs, data = wavfile.read("ambiente.wav")   # <-- poné tu archivo aquí

# Si el archivo tiene 2 canales (stereo), uso solo uno
if data.ndim > 1:
    data = data[:,0]

# Vector de tiempo
tt_audio = np.arange(len(data)) / fs

# Graficar señal
plt.figure(figsize=(10,4))
plt.plot(tt_audio, data, label="Señal WAV")
plt.title("Señal de audio")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Calcular energía (sumatoria |x|^2 * Ts)
ts_audio = 1/fs
energia = np.sum(np.abs(data)**2) * ts_audio
print(f"Energía de la señal de audio = {energia:.2f}")
