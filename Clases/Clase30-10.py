# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 14:24:17 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import matplotlib.patches as patches
from pytc2.sistemas_lineales import plot_plantilla
from scipy.interpolate import CubicSpline


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
WS = [0.1, 35.7] # comienzo y fin de la banda de stop (corresponde a la atenuacion minima y maxima) -- mas grande que la de paso

frecuencias = np.sort(np.concatenate(((0,fs/2), WP, WS)))  #frecuencia va de cero a Nyquist
deseado = [0,0,1,1,0,0]  #Puntos de mi respuesta, respecto a frecuencias(en 0 va 0, en 0.1 va 0, etc.)
cant_coef = 2000 # cant de coeficientes par
retardo = (cant_coef - 1)//2

#Método ventana
fir_win_rect = sig.firwin2(numtaps = cant_coef, freq = frecuencias, gain = deseado, window='boxcar', nfreqs = int((np.ceil(np.sqrt(cant_coef))*2)**2) - 1, fs = fs) #Esto es un filtro tipo 2

#Otro método (cuadrados mínimos)
cant_coef_ls = 2001 # cant de coeficientes impar
retardo_ls = (cant_coef_ls - 1)//2
fir_win_ls = sig.firls(numtaps = cant_coef_ls, bands = frecuencias, desired = deseado, fs = fs) #Esto es un filtro tipo 2


#Método Remez
cant_coef_rem = 3001 # cant de coeficientes impar
retardo_rem = (cant_coef_rem - 1)//2
deseado_rem = [0,1,0]
fir_win_rem = sig.remez(numtaps = cant_coef_rem, bands = frecuencias, desired = deseado_rem, fs = fs) #Esto es un filtro tipo 2


w, h = sig.freqz(b = fir_win_rem, fs=fs, worN=np.logspace(-2,1.9,3000))

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
plot_plantilla(filter_type = 'bandpass' , fpass = WP, ripple = alpha_p*2 , fstop = WS, attenuation = alpha_s*2, fs = fs)
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
   
    resul1_mediana=sig.medfilt(ecg_one_lead[zoom_region], kernel_size=201)#en fir tiene que ser impar para tener retardo entero
    estimacion=sig.medfilt(resul1_mediana, kernel_size=601)
    
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')
    plt.plot(zoom_region, estimacion, label='Estimacion')
   
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
    
    
           
    plt.show()
    

#%%
#SPLINES


# --- Seleccionamos la región entre P y Q ---
maximos = mat_struct['qrs_detections'].flatten() - 60  # posiciones de interés (60 muestras antes del R)

# --- Eje de tiempo asociado (en muestras o en segundos si conocés fs) ---
x = np.arange(len(maximos))  # vector de 0 a N-1

# --- Señal ECG ---
ecg = ecg_one_lead.flatten()

# --- Obtener valores del ECG en los puntos de interés ---
valores_ecg = ecg[maximos.astype(int)]  # valores de la señal en los máximos desplazados

# --- Interpolación con spline cúbico ---
# Creamos el spline con puntos (maximos, valores_ecg)
interpol = CubicSpline(maximos, valores_ecg)

# --- NUEVO: evaluamos el spline en TODO el eje del ECG ---
x_full = np.arange(len(ecg))        # mismo eje que la señal completa
baseline = interpol(x_full)         # spline evaluado en todas las muestras

# --- ECG corregido (sin línea de base) ---
ecg_corr = ecg - baseline

# --- GRAFICAMOS ---
plt.figure(figsize=(12,5))
plt.plot(ecg, label='ECG original', alpha=0.6)
plt.plot(baseline, 'k--', linewidth=2, label='Spline (línea de base)')
plt.plot(ecg_corr, color='g', label='ECG corregido')
plt.plot(maximos, valores_ecg, 'ro', label='Puntos usados para spline')
plt.title('Spline cúbico del ECG usando puntos de máximos')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.show()
    
#%%
# Filtro adaptado

# Ahora busco detectar los latidos

patron = mat_struct['qrs_pattern1'].flatten()
patron_util = patron - np.mean(patron)

ecg_detection = sig.lfilter(b = patron_util, a = 1, x = ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead, label = 'ECG raw')
plt.plot(np.abs(ecg_detection), label = 'Detecciones')
plt.legend()
plt.show()


# Normalizo ambas señales para verlas en la misma escala (por el desvío estándar)
ecg_detection_abs = np.abs(ecg_detection)

plt.figure()
plt.plot(ecg_one_lead/np.std(ecg_one_lead), label='ECG raw')
plt.plot(ecg_detection_abs/np.std(ecg_detection_abs), label='Detecciones')
plt.legend()
plt.show()


ecg_det_norm = ecg_detection_abs/np.std(ecg_detection_abs)

#Hago filtro pasa bajos primero
ecg_det_norm_lp = sig.lfilter(b = np.ones(111), a = 1, x = ecg_det_norm)[50:]

plt.figure()
plt.plot(ecg_one_lead/np.std(ecg_one_lead), label='ECG raw')
plt.plot(ecg_det_norm/np.std(ecg_det_norm), label='Detecciones sin filtrar')
plt.plot(ecg_det_norm_lp/np.std(ecg_det_norm_lp), label='Detecciones filtrada')
plt.legend()
plt.show()

# No me conviene trabajar sobre la señal filtrada (trabajo sobre ecg_det_norm)

mis_qrs, _ = sig.find_peaks(ecg_det_norm, height = 1., distance = 300)

delay = (len(patron_util) - 1) // 2
mis_qrs_corr = mis_qrs - delay


#%%
# Matriz de confusión

qrs_det = mat_struct['qrs_detections'].flatten()

def matriz_confusion_qrs(mis_qrs, qrs_det, tolerancia_ms=150, fs=1000):
    """
    Calcula matriz de confusión para detecciones QRS usando solo NumPy y SciPy
    
    Parámetros:
    - mis_qrs: array con tiempos de tus detecciones (muestras)
    - qrs_det: array con tiempos de referencia (muestras)  
    - tolerancia_ms: tolerancia en milisegundos (default 150ms)
    - fs: frecuencia de muestreo (default 360 Hz)
    """
    
    # Convertir a arrays numpy
    mis_qrs = np.array(mis_qrs)
    qrs_det = np.array(qrs_det)
    
    # Convertir tolerancia a muestras
    tolerancia_muestras = tolerancia_ms * fs / 1000
    
    # Inicializar contadores
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    
    # Arrays para marcar detecciones ya emparejadas
    mis_qrs_emparejados = np.zeros(len(mis_qrs), dtype=bool)
    qrs_det_emparejados = np.zeros(len(qrs_det), dtype=bool)
    
    # Encontrar True Positives (detecciones que coinciden dentro de la tolerancia)
    for i, det in enumerate(mis_qrs):
        diferencias = np.abs(qrs_det - det)
        min_diff_idx = np.argmin(diferencias)
        min_diff = diferencias[min_diff_idx]
        
        if min_diff <= tolerancia_muestras and not qrs_det_emparejados[min_diff_idx]:
            TP += 1
            mis_qrs_emparejados[i] = True
            qrs_det_emparejados[min_diff_idx] = True
    
    # False Positives (tus detecciones no emparejadas)
    tp_idx = np.where(mis_qrs_emparejados)[0]
    fp_idx = np.where(~mis_qrs_emparejados)[0]
    FP = np.sum(~mis_qrs_emparejados)
    
    # False Negatives (detecciones de referencia no emparejadas)
    fn_idx = np.where(~qrs_det_emparejados)[0]
    FN = np.sum(~qrs_det_emparejados)
    
    # Construir matriz de confusión
    matriz = np.array([
        [TP, FP],
        [FN, 0]  # TN generalmente no aplica en detección de eventos
    ])
    
    return matriz, TP, FP, FN, tp_idx, fp_idx, fn_idx

# Ejemplo de uso

matriz, tp, fp, fn, tp_idx, fp_idx, fn_idx = matriz_confusion_qrs(mis_qrs, qrs_det)

print("Matriz de Confusión:")
print(f"           Predicho")
print(f"           Sí    No")
print(f"Real Sí:  [{tp:2d}   {fn:2d}]")
print(f"Real No:  [{fp:2d}    - ]")
print(f"\nTP: {tp}, FP: {fp}, FN: {fn}")

# Calcular métricas de performance
if tp + fp > 0:
    precision = tp / (tp + fp)
else:
    precision = 0

if tp + fn > 0:
    recall = tp / (tp + fn)
else:
    recall = 0

if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

print(f"\nMétricas:")
print(f"Precisión: {precision:.3f}")
print(f"Sensibilidad: {recall:.3f}")
print(f"F1-score: {f1_score:.3f}")

plt.figure(figsize=(15,5))
plt.plot(ecg_one_lead, label="ECG raw", linewidth=1)
plt.plot(mis_qrs[tp_idx], ecg_one_lead[mis_qrs[tp_idx]], 'go', label="TP (correctas)")
plt.plot(mis_qrs[fp_idx], ecg_one_lead[mis_qrs[fp_idx]], 'ro', label="FP (falsos positivos)")
plt.plot(qrs_det[fn_idx], ecg_one_lead[qrs_det[fn_idx]], 'mo', label="FN (no detectadas)")

plt.legend()
plt.title("Detecciones QRS: TP / FP / FN")
plt.xlabel("Muestras")
plt.ylabel("Amplitud ECG")
plt.grid(True)
plt.show()


# ======= Otra cosa que hizo ========

qrs_mat = np.array([ecg_one_lead[ii-60 : ii+60] for ii in mis_qrs_corr])

# Normalización restando la media de cada latido
qrs_mat = qrs_mat - np.mean(qrs_mat, axis=1).reshape((-1, 1))


plt.plot(qrs_mat.transpose())


#Veo los latidos alineados donde los encontramos
# si promedio, la señal resultante va a pasar por donde hay más valores
#pero como hay dos formas de latidos distintas, va a dar algo ni (el promedio no será
# capaz de representar a todos los latidos)

# Si quiero sacar un latido normal promedio, debo separar los dos tipos de latidos según
# su morfología.


#%%

mis_qrs, promin = sig.find_peaks(ecg_det_norm, height = 1., distance = 300, 
                                 prominence = (None, 6))
# Filtra los qrs que eran superiores a 6 (tendré menos mis_qrs)

delay = (len(patron_util) - 1) // 2
mis_qrs_corr = mis_qrs - delay

qrs_mat = np.array([ecg_one_lead[ii-60 : ii+60] for ii in mis_qrs_corr])

# Normalización restando la media de cada latido
qrs_mat = qrs_mat - np.mean(qrs_mat, axis=1).reshape((-1, 1))


plt.plot(qrs_mat.transpose())


# Para refinar las detecciones --> Algortimo de Woody (?)
# Toma todo el paquete de señales, estima un primer patrón promedio
# Calcula un latido medio, y calcula la correlación cruzada entre este latido medio
# y cada realización.
# El máximo se encontrará donde haya mayor solapamiento entre las realizaciones y el
# patrón. 

# El algoritmo de Woody lleva a la convergencia de todos los latidos (alinea realizaciones)
# Después, promedio. Este algoritmo me da un mejor promedio, y puedo mejorar mi detección en general.


## Vuelvo a hacer lo mismo pero con el Algoritmo de Woody antes
mis_qrs, promin = sig.find_peaks(ecg_detection_abs, height=1., distance=300,
                                 prominence=(None, 6), width=(19, 23))

qrs_mat = np.array([ ecg_one_lead[ii-60:ii+60] for ii in mis_qrs ])

qrs_mat = qrs_mat - np.mean(qrs_mat, axis = 1).reshape((-1,1))

plt.plot(qrs_mat.transpose())


# %%


# import numpy as np
# from scipy.signal import correlate
# from scipy.optimize import minimize_scalar

# def woody_alignment(ecg_signal, detections, window_before=100, window_after=150, max_shift=50, fs=1000):
#     """
#     Algoritmo de Woody para refinar detecciones de QRS mediante alineación por correlación cruzada
    
#     Parámetros:
#     - ecg_signal: señal de ECG completa
#     - detections: array con las detecciones iniciales (en muestras)
#     - window_before: muestras antes del punto de detección para extraer el template
#     - window_after: muestras después del punto de detección para extraer el template  
#     - max_shift: máximo desplazamiento permitido en muestras
#     - fs: frecuencia de muestreo
    
#     Retorna:
#     - refined_detections: detecciones refinadas
#     - templates: templates utilizados
#     - shifts: desplazamientos aplicados a cada detección
#     """
    
#     detections = np.array(detections, dtype=int)
#     refined_detections = detections.copy()
#     shifts = np.zeros(len(detections), dtype=int)
    
#     # Paso 1: Crear template inicial (promedio de los primeros N latidos)
#     n_template = min(10, len(detections))
#     template_segments = []
    
#     for i in range(n_template):
#         start = max(0, detections[i] - window_before)
#         end = min(len(ecg_signal), detections[i] + window_after)
#         segment = ecg_signal[start:end]
#         template_segments.append(segment)
    
#     # Asegurar que todos los segmentos tengan la misma longitud
#     min_length = min(len(seg) for seg in template_segments)
#     template_segments = [seg[:min_length] for seg in template_segments]
    
#     template = np.mean(template_segments, axis=0)
    
#     # Paso 2: Iterar el proceso de alineación (máximo 5 iteraciones)
#     for iteration in range(5):
#         print(f"Iteración Woody {iteration + 1}")
#         total_shift = 0
        
#         for i, det in enumerate(detections):
#             # Extraer segmento alrededor de la detección actual
#             start = max(0, det - window_before - max_shift)
#             end = min(len(ecg_signal), det + window_after + max_shift)
#             segment = ecg_signal[start:end]
            
#             # Calcular correlación cruzada con el template
#             correlation = correlate(segment, template, mode='valid', method='auto')
#             correlation = correlate(segment, template, mode='full', method='auto')
            
#             # Encontrar el desplazamiento óptimo
#             lag = np.argmax(correlation) - (len(segment) - len(template))
            
#             # Limitar el desplazamiento al máximo permitido
#             lag = np.clip(lag, -max_shift, max_shift)
            
#             # Aplicar el desplazamiento
#             new_detection = det + lag
            
#             # Verificar que la nueva detección esté dentro de los límites
#             if (new_detection - window_before >= 0 and 
#                 new_detection + window_after < len(ecg_signal)):
#                 refined_detections[i] = new_detection
#                 shifts[i] = lag
#                 total_shift += abs(lag)
        
#         # Paso 3: Actualizar el template con las nuevas posiciones
#         template_segments = []
#         for det in refined_detections:
#             start = max(0, det - window_before)
#             end = min(len(ecg_signal), det + window_after)
#             segment = ecg_signal[start:end]
#             if len(segment) == len(template):  # Solo usar segmentos de longitud correcta
#                 template_segments.append(segment)
        
#         if template_segments:
#             new_template = np.mean(template_segments, axis=0)
#             template_change = np.mean(np.abs(new_template - template))
#             template = new_template
            
#             # Criterio de convergencia
# if template_change < 0.01 or total_shift < len(detections):
#                 print(f"Convergencia alcanzada en iteración {iteration + 1}")
#                 break
    
#     return refined_detections, template, shifts

# # Supongamos que tienes:
# # ecg_signal: tu señal de ECG
# # mis_detecciones: tus detecciones iniciales
# # qrs_referencia: detecciones de referencia (si las tienes)

# # Aplicar Woody
# detecciones_refinadas, template_final, desplazamientos = woody_alignment(
# ecg_signal=ecg_one_lead,
# detections=mis_qrs,
# window_before=100,
# window_after=100,
# max_shift=50,
# )

# print(f"Detecciones originales: {len(mis_qrs)}")
# print(f"Detecciones refinadas: {len(detecciones_refinadas)}")
# print(f"Desplazamientos aplicados: {np.sum(np.abs(desplazamientos))} muestras totales")


# qrs_mat = np.array([ ecg_one_lead[ii-60:ii+60] for ii in detecciones_refinadas ])

# qrs_mat = qrs_mat - np.mean(qrs_mat, axis = 1).reshape((-1,1))

# plt.figure(2)
# plt.plot(qrs_mat.transpose())