# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:02:18 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

# Plantilla de diseño
fs = 1000 # Hz
ws1 = 0.1 # Hz
wp1 = 0.8 # Hz
wp2 = 35 # Hz
ws2 = 40 # Hz
WP = [wp1, wp2] # comienzo y fin de la banda de paso
WS = [ws1, ws2] # comienzo y fin de la banda de stop (corresponde a la atenuacion minima y maxima) -- mas grande que la de paso
 
nyq_frec = fs/2
ripple = 1/2 # dB
atenuacion = 40/2 # dB
  
# plantilla normalizada a Nyquist en dB
frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ])/nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
 
# convertimos a veces para las funciones de diseño
gains = 10**(gains/20)

#%%
##################
# Lectura de ECG #
##################

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
cant_muestras = len(ecg_one_lead)

#%%

#Diseño cuatro filtros IIR (aproximaciones de módulo):

f_aprox_butter = 'butter'
f_aprox_ch1 = 'cheby1'
f_aprox_ch2 = 'cheby2'
f_aprox_cauer = 'cauer'

mi_sos_butter = sig.iirdesign(wp=WP, ws=WS, gpass=ripple, gstop=atenuacion, analog=False, ftype=f_aprox_butter, output='sos', fs=fs)
mi_sos_cheby1 = sig.iirdesign(wp=WP, ws=WS, gpass=ripple, gstop=atenuacion, analog=False, ftype=f_aprox_ch1, output='sos', fs=fs)
mi_sos_cheby2 = sig.iirdesign(wp=WP, ws=WS, gpass=ripple, gstop=atenuacion, analog=False, ftype=f_aprox_ch2, output='sos', fs=fs)
mi_sos_cauer = sig.iirdesign(wp=WP, ws=WS, gpass=ripple, gstop=atenuacion, analog=False, ftype=f_aprox_cauer, output='sos', fs=fs)


def plot_sos(sos, fs, name="Filtro IIR",
             filter_type=None, fpass=None, ripple=None,
             fstop=None, attenuation=None):
 
    # RESPUESTA EN FRECUENCIA
    w, h = sig.freqz_sos(sos=sos, fs=fs, worN=np.logspace(-2, 1.9, 2000))

    # FASE Y RETARDO DE GRUPO
    phase = np.unwrap(np.angle(h))
    w_rad = w / (fs/2) * np.pi
    gd = -np.diff(phase) / np.diff(w_rad)

    # GRÁFICAS
    plt.figure(figsize=(12, 12))

    # MAGNITUD
    plt.subplot(4, 1, 1)
    plt.plot(w, 20*np.log10(np.abs(h)), label=name)
    plt.title(f'Respuesta en Magnitud — {name}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    plt.grid(True, ls=':')
    plt.legend()

    # FASE
    plt.subplot(4, 1, 2)
    plt.plot(w, phase, label=name)
    plt.title(f'Fase — {name}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.grid(True, ls=':')
    plt.legend()

    # RETARDO DE GRUPO
    plt.subplot(4, 1, 3)
    plt.plot(w[:-1], gd, label=name)
    plt.title(f'Retardo de Grupo — {name}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [muestras]')
    plt.grid(True, ls=':')
    plt.legend()

    plt.tight_layout()
    plt.show()

    
plot_sos(sos=mi_sos_butter, fs=fs, name='Butter', filter_type='bandpass', fpass=WP, ripple=ripple*2, fstop=WS, attenuation=atenuacion*2)
plot_sos(sos=mi_sos_cheby1, fs=fs, name='Cheby 1', filter_type='bandpass', fpass=WP, ripple=ripple*2, fstop=WS, attenuation=atenuacion*2)
plot_sos(sos=mi_sos_cheby2, fs=fs, name='Cheby 2', filter_type='bandpass', fpass=WP, ripple=ripple*2, fstop=WS, attenuation=atenuacion*2)
plot_sos(sos=mi_sos_cauer, fs=fs, name='Cauer', filter_type='bandpass', fpass=WP, ripple=ripple*2, fstop=WS, attenuation=atenuacion*2)
#%%
# Diseño de filtros FIR

frecuencias = np.sort(np.concatenate(((0,fs/2), WP, WS)))  #frecuencia va de cero a Nyquist
deseado = [0,0,1,1,0,0]  #Puntos de mi respuesta, respecto a frecuencias(en 0 va 0, en 0.1 va 0, etc.)


#Método ventana
cant_coef_win = 1501
retardo_win = (cant_coef_win - 1)//2
fir_win_rect = sig.firwin2(numtaps = cant_coef_win, freq = frecuencias, gain = deseado, window='boxcar', nfreqs = int((np.ceil(np.sqrt(cant_coef_win))*2)**2) - 1, fs = fs) #Esto es un filtro tipo 2

#Otro método (cuadrados mínimos)
cant_coef_ls = 999 # cant de coeficientes impar
retardo_ls = (cant_coef_ls - 1)//2
frecuencias_ls = [0, ws1, wp1, wp2, ws2, fs/2]
weights_ls = [3, 0.2, 3]
fir_ls = sig.firls(numtaps=cant_coef_ls,
    bands=frecuencias_ls, desired=deseado, weight=weights_ls, fs=fs)

#Método Remez
cant_coef_rem = 999 # cant de coeficientes impar
retardo_rem = (cant_coef_rem - 1)//2
frecuencias_rem = [0, ws1, wp1, wp2, ws2, fs/2]
deseado_rem = [0, 1, 0]
weights_rem = [20, 1, 20]   # stopband más pesado
fir_rem = sig.remez(numtaps=cant_coef_rem, bands=frecuencias_rem, desired=deseado_rem, weight=weights_rem, fs=fs)

def plot_filtro_fir(b, fs, name="Filtro FIR",
                filter_type=None, fpass=None, ripple=None, 
                fstop=None, attenuation=None):
  
    w, h = sig.freqz(b=b, fs=fs, worN=np.logspace(-2, 1.9, 3000))

    phase = np.unwrap(np.angle(h))
    w_rad = w / (fs/2) * np.pi
    gd = -np.diff(phase) / np.diff(w_rad)

    #   GRÁFICAS
    plt.figure(figsize=(12, 10))

    # === MAGNITUD ===
    plt.subplot(3, 1, 1)
    plt.plot(w, 20*np.log10(np.abs(h)), label=name)
    if filter_type is not None:
        plot_plantilla(filter_type, fpass, ripple, fstop, attenuation, fs)
    plt.title(f'Respuesta en Magnitud — {name}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # === FASE ===
    plt.subplot(3, 1, 2)
    plt.plot(w, phase, label=name)
    plt.title(f'Fase — {name}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # === RETARDO DE GRUPO ===
    plt.subplot(3, 1, 3)
    plt.plot(w[:-1], gd, label=name)
    plt.title(f'Retardo de Grupo — {name}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [muestras]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_filtro_fir(b=fir_win_rect, fs=fs, name='Ventana rectangular', filter_type='bandpass', fpass=WP, ripple=ripple*2, fstop=WS, attenuation=atenuacion*2)
plot_filtro_fir(b=fir_ls, fs=fs, name='Cuadrados mínimos', filter_type='bandpass', fpass=WP, ripple=ripple*2, fstop=WS, attenuation=atenuacion*2)
plot_filtro_fir(b=fir_rem, fs=fs, name='Parks-McClellan', filter_type='bandpass', fpass=WP, ripple=ripple*2, fstop=WS, attenuation=atenuacion*2)

#%%

ecg_filt_win = sig.lfilter(b = fir_win_rect, a = 1, x = ecg_one_lead)
ecg_filt_ls = sig.lfilter(b = fir_ls, a = 1, x = ecg_one_lead)
ecg_filt_rem = sig.lfilter(b = fir_rem, a = 1, x = ecg_one_lead)

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
plt.plot(ecg_filt_win[:50000], label = 'FIR ventana')
plt.plot(ecg_filt_ls[:50000], label = 'FIR cuadrados mínimos')
plt.plot(ecg_filt_rem[:50000], label = 'FIR Parks-McClellan')

plt.legend()


#%%
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
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo_win], label='FIR ventana')
    plt.plot(zoom_region, ecg_filt_ls[zoom_region + retardo_ls], label='FIR cuadrados mínimos')
    plt.plot(zoom_region, ecg_filt_rem[zoom_region + retardo_rem], label='FIR Parks-McClellan')
    plt.plot(zoom_region, ecg_filt_cheby1[zoom_region], label='Cheby 1')
    plt.plot(zoom_region, ecg_filt_cheby2[zoom_region], label='Cheby 2')
    plt.plot(zoom_region, ecg_filt_cauer[zoom_region], label='Cauer')
    
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
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo_win], label='FIR Window')
    plt.plot(zoom_region, ecg_filt_ls[zoom_region + retardo_ls], label='FIR cuadrados mínimos')
    plt.plot(zoom_region, ecg_filt_rem[zoom_region + retardo_rem], label='FIR Parks-McClellan')
    plt.plot(zoom_region, ecg_filt_cheby1[zoom_region], label='Cheby 1')
    plt.plot(zoom_region, ecg_filt_cheby2[zoom_region], label='Cheby 2')
    plt.plot(zoom_region, ecg_filt_cauer[zoom_region], label='Cauer')
    
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
    
    plt.show()

