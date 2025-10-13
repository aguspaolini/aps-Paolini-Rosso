# -*- coding: utf-8 -*-
"""
Created on Fri Aug  14 17:05:23 2025

@author: aguspaolini
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
from scipy.io import wavfile
from scipy.signal import correlate

#%%                                  Punto 1

fs = 100000  # Frecuencia de muestreo [Hz]
ts = 1/fs     # Tiempo entre muestras [s]
nn = 500      # Cantidad de muestras → duración total 5 ms
tt = np.arange(nn)*ts   # Vector de tiempo

# Función para graficar y mostrar Ts, N y energía
def grafico(x, y, titulo):
    N = len(y)                     # Cantidad de muestras
    Ts = x[1]-x[0]                 # Tiempo entre muestras
    E = np.sum(np.abs(y)**2) * Ts  # Energía de la señal
    P = (1/(N*Ts)) * np.sum(np.abs(y)**2) * Ts   
   
    
    plt.figure(figsize=(8,4))
    line_hdls = plt.plot(x, y, 'go', label='Señal', markersize=2)
    plt.title(titulo)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.grid(True)
    
    # Información en la leyenda
    if titulo == 'Pulso rectangular 10 ms':
        info_text = f"ts = {Ts*1e6:.2f} µs, N = {N}, Energía = {E:.4f} J"
        plt.legend(line_hdls, [info_text], loc='center right')
    else:
        info_text = f"ts = {Ts*1e6:.2f} µs, N = {N}, Potencia = {P:.4f} W"
        plt.legend(line_hdls, [info_text], loc='upper right')
        plt.xlim(0, 5e-3)
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
m = 0.7        # índice de modulación (0 < m <= 1)
s_moduladora = np.sin(2*np.pi*ff_mod*tt)
sen_3 = (1 + m*s_moduladora) * sen_1
env_sup = (1 + m*s_moduladora)
env_inf = -(1 + m*s_moduladora)

plt.figure(figsize=(8,4))
P = (1/(nn*ts)) * np.sum(np.abs(sen_3)**2) * ts  
N = len(sen_3)                     # Cantidad de muestras
Ts = tt[1]-tt[0] 

# Graficar
line1, = plt.plot(tt, sen_3, 'go', label='S1 modulada en amplitud', markersize=2)
line2, = plt.plot(tt, env_sup, 'r--', label='Envolvente superior')
line3, = plt.plot(tt, env_inf, 'y--', label='Envolvente inferior')

# Info extra
info_text = f"ts = {Ts*1e6:.2f} µs, N = {N}, Potencia = {P:.4f} W"

# Primera leyenda (las señales)
legend1 = plt.legend([line1, line2, line3],
                     [line1.get_label(), line2.get_label(), line3.get_label()],
                     loc='upper right')
plt.gca().add_artist(legend1)  # fijar esta leyenda

# Segunda leyenda (info aparte)
plt.legend([line1], [info_text], loc='lower right')

plt.title('Modulación')
plt.xlabel('Tiempo [s]')
plt.xlim(0, 5e-3)
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.tight_layout()
plt.show()



def recorte_potencia(signal, porcentaje=0.75):
      
    # Potencia original (discreta, promedio)
    P_orig = np.mean(np.abs(signal)**2)
    P_obj = porcentaje * P_orig

    # Máxima amplitud de la señal
    max_amp = np.max(np.abs(signal))
    umbral = max_amp

    # Búsqueda iterativa del umbral
    for i in np.linspace(max_amp, 0, 2000):
        rec = np.clip(signal, -i, i)
        P_rec = np.mean(np.abs(rec)**2)   # potencia promedio, no energía
        if P_rec <= P_obj:
            umbral = i
            break

    signal_rec = np.clip(signal, -umbral, umbral)
    return signal_rec, umbral



def recorte_por_amplitud(x, factor=0.75):
    A = np.max(np.abs(x))
    u = factor * A
    return np.clip(x, -u, u), u

sen_4, umbral = recorte_potencia(sen_1,0.75)
sen_4_modif, umbral_modif = recorte_por_amplitud(sen_1,0.75)
grafico(tt, sen_4, 'Señal recortada en amplitud al 75% de la potencia')
grafico(tt, sen_4_modif, 'Señal recortada al 75% de su amplitud')


# Señal 5: cuadrada 4 kHz
ff_2 = 4000  # Hz
sen_5 = square(2*np.pi*ff_2*tt)
grafico(tt, sen_5, 'Señal cuadrada 4 Khz')


# Señal 6: pulso rectangular 10 ms

nn_pulso = int(15e-3*fs)   # número total de muestras dado 15 ms
tt_pulso = np.arange(nn_pulso)*ts   # vector de tiempo

sen_6 = np.zeros_like(tt_pulso)    # creo vector de igual longitud
n_pulso = int(10e-3*fs)            # duración del pulso = 10 ms
sen_6[:n_pulso] = 1                # vale 1 en esos 10 ms

grafico(tt_pulso, sen_6, 'Pulso rectangular 10 ms')


#%%                           Verifico ortogonalidad


def prod_interno(s1, s2, ts, n1, n2, tol=1e-10):
    
    # Producto interno con np.dot
    p_int = np.dot(s1, s2) * ts

    if np.abs(p_int) < tol:
        print(f'Las señales {n1} y {n2} son ortogonales.')
    else:
        print(f'Las señales {n1} y {n2} NO son ortogonales.')


# Verificar ortogonalidad
prod_interno(sen_1, sen_2, ts, 'S1', 'S2', 1e-10)
prod_interno(sen_1, sen_3, ts, 'S1', 'S3', 1e-10)
prod_interno(sen_1, sen_4, ts, 'S1', 'S4', 1e-10)
prod_interno(sen_1, sen_5, ts, 'S1', 'S5', 1e-10)
prod_interno(sen_1, sen_6[:len(sen_1)], ts, 'S1', 'Pulso', 1e-10)

#%%                       Punto 3
def correlacion(x, y, titulo, ts):
    corr = correlate(x, y, mode='full', method='direct') * ts
    lags = np.arange(-len(x)+1, len(x))
    
    
    plt.figure(figsize=(8,4))
    plt.plot(lags, corr, 'm')
    plt.title(titulo)
    plt.xlabel('Desplazamiento temporal [muestras]')
    plt.ylabel('Correlación')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Autocorrelación de S1
correlacion(sen_1, sen_1, 'Autocorrelación S1 (2 kHz)', ts)

# Correlación entre S1 y las demás
correlacion(sen_1, sen_2, 'Correlación S1-S2', ts)
correlacion(sen_1, sen_3, 'Correlación S1-S3', ts)
correlacion(sen_1, sen_4, 'Correlación S1-S4', ts)
correlacion(sen_1, sen_4_modif, 'Correlación S1-S4 modif', ts)
correlacion(sen_1, sen_5, 'Correlación S1-S5', ts)
correlacion(sen_1, sen_6[:len(sen_1)], 'Correlación S1-Pulso', ts)


# --- Señal cuadrada a 2 kHz ---
sen_5 = square(2*np.pi*2000*tt)  # misma frecuencia que sen_1
grafico(tt, sen_5, 'Señal cuadrada 2 kHz')
correlacion(sen_1, sen_5, 'Correlación: S1 vs Cuadrada 2 kHz', ts)

# Señal 1: senoidal de 2000 Hz (extendida a 15 ms)
f_sine = 2000
sen_1_largo = np.sin(2 * np.pi * f_sine * tt_pulso)
correlacion(sen_1_largo, sen_6, 'Correlación S1-Pulso', ts)

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


#%% ==================== Bonus 4: Procesar un WAV ====================

# Leer archivo wav
fs, data = wavfile.read("ambiente.wav")   # <-- poné tu archivo aquí

# Si el archivo tiene 2 canales (stereo), uso solo uno
if data.ndim > 1:
    data = data[:,0]

# Vector de tiempo
tt_audio = np.arange(len(data)) / fs

# Graficar señal
plt.figure(figsize=(10,4))
plt.plot(tt_audio, data,'bo', label="Señal WAV", markersize=2)
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
