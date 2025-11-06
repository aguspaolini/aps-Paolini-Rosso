# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 14:24:17 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import matplotlib.patches as patches

# Plantilla de diseño
fs=1000 # Hz
WP = [0.8, 35] # comienzo y fin de la banda de paso
WS = [0.1, 40] # comienzo y fin de la banda de stop (corresponde a la atenuacion minima y maxima) -- mas grande que la de paso


# alpha_p = 1  # atenuacion
# alpha_s = 40

alpha_p = 1/2  # atenuacion para sosfiltfilt (paso dos veces por el filtro)
alpha_s = 40/2

# Aproximacion de fase
# f_aprox = 'bessel'

#Diseño cuatro filtros:
# Aproximaciones de modulo
f_aprox = 'butter'
    
# Diseño del filtro Butterworth analogico
mi_sos_butter = sig.iirdesign(wp=WP, ws=WS, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)
# mi_sos = mi_sos_butter


f_aprox = 'cheby1'
mi_sos_cheby1 = sig.iirdesign(wp=WP, ws=WS, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)
# mi_sos = mi_sos_cheby1

f_aprox = 'cheby2'
mi_sos_cheby2 = sig.iirdesign(wp=WP, ws=WS, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)
# mi_sos = mi_sos_cheby2

f_aprox = 'cauer'
mi_sos_cauer = sig.iirdesign(wp=WP, ws=WS, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)
mi_sos = mi_sos_cauer


#%%

# Respuesta en frecuencia
# w, h = signal.freqs(b, a, worN=np.logspace(1, 6, 1000)) # espacio logaritmicamente espaciado
#w, h = signal.freqs(mi_sos) # calcula la respuesta en frecuencia del filtro
#w, h = signal.freqz_sos(sos=mi_sos, fs=fs) # agregamos fs=fs para que el w salga en hz
w, h = sig.freqz_sos(sos=mi_sos, fs=fs, worN=np.logspace(-2,1.9,1000))

# --- Cálculo de fase y retardo de grupo ---
phase = np.unwrap(np.angle(h))
# Retardo de grupo = -dφ/dω
w_rad = w / (fs/2) *np.pi
gd = -np.diff(phase) / np.diff(w_rad)

# --- Polos y ceros ---
z, p, k = sig.sos2zpk(mi_sos)

# --- Gráficas ---
plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w, 20*np.log10(abs(h)),label=f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(3,1,2)
plt.plot(w, phase, label = f_aprox)
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(3,1,3)
plt.plot(w[:-1], gd)
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')

# Diagrama de polos y ceros

plt.figure(figsize=(8,10))
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
axes_hdl = plt.gca()

if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
axes_hdl.add_patch(unit_circle)
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#%%
   
import scipy.io as sio

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
cant_muestras = len(ecg_one_lead)

# ecg_filt_butter = sig.sosfilt(mi_sos_butter, ecg_one_lead)
# ecg_filt_cheby1 = sig.sosfilt(mi_sos_cheby1, ecg_one_lead)
# ecg_filt_cheby2 = sig.sosfilt(mi_sos_cheby2, ecg_one_lead)
# ecg_filt_cauer = sig.sosfilt(mi_sos_cauer, ecg_one_lead)

ecg_filt_butter = sig.sosfiltfilt(mi_sos_butter, ecg_one_lead)
ecg_filt_cheby1 = sig.sosfiltfilt(mi_sos_cheby1, ecg_one_lead)
ecg_filt_cheby2 = sig.sosfiltfilt(mi_sos_cheby2, ecg_one_lead)
ecg_filt_cauer = sig.sosfiltfilt(mi_sos_cauer, ecg_one_lead)

plt.figure()

plt.plot(ecg_one_lead[:50000], label = 'ECG sin filtrar')
plt.plot(ecg_filt_butter[:50000], label = 'butter')
plt.plot(ecg_filt_cheby1[:50000], label = 'cheby1')
plt.plot(ecg_filt_cheby2[:50000], label = 'cheby2')
plt.plot(ecg_filt_cauer[:50000], label = 'cauer')

plt.legend()

#%%
#==========================
# Diseño de filtros FIR
#==========================


WP = [0.8, 35] # comienzo y fin de la banda de paso
WS = [0.1, 40] # comienzo y fin de la banda de stop (corresponde a la atenuacion minima y maxima) -- mas grande que la de paso

frecuencias = np.sort(np.concatenate(((0,fs/2), WP, WS)))  #frecuencia va de cero a Nyquist
deseado = [0,0,1,1,0,0]  #Puntos de mi respuesta, respecto a frecuencias(en 0 va 0, en 0.1 va 0, etc.)
cant_coef = 2000 # cant de coeficientes par
retardo = (cant_coef - 1)//2

fir_win_rect = sig.firwin2(numtaps = cant_coef, freq = frecuencias, gain = deseado, window='boxcar', nfreqs = int((np.ceil(np.sqrt(cant_coef))*2)**2) - 1, fs = fs) #Esto es un filtro tipo 2

w, h = sig.freqz(b = fir_win_rect, fs=fs, worN=np.logspace(-2,1.9,1000))

# --- Cálculo de fase y retardo de grupo ---
phase = np.unwrap(np.angle(h))
# Retardo de grupo = -dφ/dω
w_rad = w / (fs/2) *np.pi
gd = -np.diff(phase) / np.diff(w_rad)

# --- Polos y ceros ---
# z, p, k = sig.sos2zpk(sig.tf2sos(b = fir_win_hamming, a = 1))

# --- Gráficas ---
plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w, 20*np.log10(abs(h)),label=f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(3,1,2)
plt.plot(w, phase, label = f_aprox)
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(3,1,3)
plt.plot(w[:-1], gd)
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')

# # Diagrama de polos y ceros

# plt.figure(figsize=(8,10))
# plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
# axes_hdl = plt.gca()

# if len(z) > 0:
#     plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
# plt.axhline(0, color='k', lw=0.5)
# plt.axvline(0, color='k', lw=0.5)
# unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
# axes_hdl.add_patch(unit_circle)
# plt.axis([-1.1, 1.1, -1.1, 1.1])
# plt.title('Diagrama de Polos y Ceros (plano s)')
# plt.xlabel('σ [rad/s]')
# plt.ylabel('jω [rad/s]')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()


ecg_filt_win = sig.lfilter(b = fir_win_rect, a = 1, x = ecg_one_lead)

###################################
# Regiones de interés sin ruido #
###################################
 
regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')
   
    plt.title('ECG sin ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
###################################
# Regiones de interés con ruido #
###################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='Butterworth')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
    
