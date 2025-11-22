# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 14:14:10 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.interpolate import CubicSpline
import scipy.io as sio

# ============================================================
#  CONFIGURACIÓN INICIAL
# ============================================================
fs = 1000  # Hz

# Filtros de mediana: 200 ms y 600 ms
kernel_med1 = 201   # 200 ms -> 200 + 1
kernel_med2 = 601   # 600 ms -> 600 + 1

# ============================================================
#  LECTURA DE ECG
# ============================================================
mat_struct = sio.loadmat("./ECG_TP4.mat")
ecg_one_lead = mat_struct["ecg_lead"].flatten()
qrs_det = mat_struct["qrs_detections"].flatten()
cant_muestras = len(ecg_one_lead)

# ============================================================
#  1) FILTRO DE MEDIANA
#     b^(n) = med600( med200( s[n] ) )
# ============================================================

# Aplicamos el filtro de mediana de 200 ms
med_200 = sig.medfilt(ecg_one_lead, kernel_size=kernel_med1)

# Luego el filtro de 600 ms sobre la salida anterior
baseline_med = sig.medfilt(med_200, kernel_size=kernel_med2)

# ECG filtrado (estimación sin baseline)
ecg_med_corr = ecg_one_lead - baseline_med

plt.figure(figsize=(12,4))
plt.plot(ecg_one_lead, label="ECG original", alpha=0.6)
plt.plot(baseline_med, 'k', linewidth=2, label="Baseline estimada (medianas)")
plt.plot(ecg_med_corr, 'g', label="ECG corregido")
plt.legend()
plt.title("Método 1: Filtro de mediana")
plt.show()

# ============================================================
#  2) INTERPOLACIÓN SPLINE CÚBICO
#     Usando puntos en el segmento PQ
# ============================================================

# Aprox. 60 ms antes del QRS
n0 = 60  
mi = qrs_det - n0
mi = mi.astype(int)

# Asegurar que están dentro del rango
mi = mi[(mi > 0) & (mi < len(ecg_one_lead))]

valores_ecg = ecg_one_lead[mi]

# Construcción del spline cúbico
spl = CubicSpline(mi, valores_ecg)

# Evaluación sobre TODA la señal
x_full = np.arange(len(ecg_one_lead))
baseline_spline = spl(x_full)

ecg_spline_corr = ecg_one_lead - baseline_spline

plt.figure(figsize=(12,4))
plt.plot(ecg_one_lead, label="ECG original", alpha=0.5)
plt.plot(baseline_spline, 'k--', linewidth=2, label="Spline (baseline)")
plt.plot(ecg_spline_corr, 'g', label="ECG corregido")
plt.plot(mi, valores_ecg, 'ro', label="Puntos PQ usados")
plt.legend()
plt.title("Método 2: Interpolación spline cúbico")
plt.show()

# ============================================================
#  3) FILTRO ADAPTADO (MATCHED FILTER)
# ============================================================

# ----------- EXPLICACIÓN (requerida por la consigna) ----------
"""
Conceptualmente, un filtro adaptado correlaciona la señal ECG con un patrón
ideal del QRS. Cuando el patrón se alinea con un QRS real, la correlación se
vuelve máxima, produciendo un pico que permite detectar los latidos.

Ventajas:
- Muy buena detección en presencia de ruido blanco.
- Realza las formas similares al patrón QRS.

Limitaciones:
- Depende fuertemente del patrón: si los QRS cambian, pierde performance.
- No es robusto a latidos anormales o variabilidad morfológica.
- Produce un retardo que debe corregirse.
"""

# ============================================================
# 3.a) Filtro adaptado
# ============================================================
patron = mat_struct["qrs_pattern1"].flatten()
patron = patron - np.mean(patron)     # remover DC

# Correlación implementada como filtrado FIR
ecg_detection = sig.lfilter(patron, 1, ecg_one_lead)
ecg_det_abs = np.abs(ecg_detection)

# Normalización
ecg_det_norm = ecg_det_abs / np.std(ecg_det_abs)

# Corrección del retardo del FIR
delay = (len(patron) - 1) // 2

# ============================================================
# 3.b) Detección de picos
# ============================================================
mis_qrs, _ = sig.find_peaks(ecg_det_norm, height=1.0, distance=300)
mis_qrs_corr = mis_qrs - delay
mis_qrs_corr = mis_qrs_corr[mis_qrs_corr > 0]

plt.figure(figsize=(12,4))
plt.plot(ecg_one_lead, label="ECG")
plt.plot(mis_qrs_corr, ecg_one_lead[mis_qrs_corr], 'ro', label="Detecciones")
plt.legend()
plt.title("Detección de QRS usando filtro adaptado")
plt.show()

# ============================================================
# 3.c) MATRIZ DE CONFUSIÓN
# ============================================================

def matriz_confusion_qrs(mis_qrs, qrs_det, tolerancia_ms=150, fs=1000):
    mis_qrs = np.array(mis_qrs)
    qrs_det = np.array(qrs_det)

    tol = tolerancia_ms * fs / 1000

    TP = 0
    FP = 0
    FN = 0

    mis_usados = np.zeros(len(mis_qrs), dtype=bool)
    ref_usados = np.zeros(len(qrs_det), dtype=bool)

    # Emparejar detecciones
    for i, det in enumerate(mis_qrs):
        diff = np.abs(qrs_det - det)
        idx = np.argmin(diff)

        if diff[idx] <= tol and not ref_usados[idx]:
            TP += 1
            mis_usados[i] = True
            ref_usados[idx] = True

    FP = np.sum(~mis_usados)
    FN = np.sum(~ref_usados)

    return TP, FP, FN, mis_usados, ref_usados

TP, FP, FN, mis_usados, ref_usados = matriz_confusion_qrs(mis_qrs_corr, qrs_det)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n=== MÉTRICAS DEL DETECTOR ===")
print(f"TP = {TP}, FP = {FP}, FN = {FN}")
print(f"Precisión (PPV) = {precision:.3f}")
print(f"Sensibilidad = {recall:.3f}")
print(f"F1-score = {f1:.3f}")

# Gráfico con TP, FP, FN
tp_idx = np.where(mis_usados)[0]
fp_idx = np.where(~mis_usados)[0]
fn_idx = np.where(~ref_usados)[0]

plt.figure(figsize=(14,5))
plt.plot(ecg_one_lead, label="ECG")
plt.plot(mis_qrs_corr[tp_idx], ecg_one_lead[mis_qrs_corr[tp_idx]], 'go', label="TP")
plt.plot(mis_qrs_corr[fp_idx], ecg_one_lead[mis_qrs_corr[fp_idx]], 'ro', label="FP")
plt.plot(qrs_det[fn_idx], ecg_one_lead[qrs_det[fn_idx]], 'mo', label="FN")
plt.legend()
plt.title("Performance del detector (Filtro adaptado)")
plt.show()



# %% ======================= BONUS: Baseline multirate ===========================
# Idea: bajar la frecuencia para eliminar detalles de alta frecuencia,
# aplicar un suavizado suave, y luego remuestrear a fs original.
# Esto funciona como un filtro pasa–bajos muy eficiente.

# Factor de diezmado (baja frecuencia)
D = 50     # elegí un valor moderado para no distorsionar demasiado

# Diezmar la señal ECG
ecg_dec = ecg_one_lead[::D]

# Suavizado en baja frecuencia (ventana grande porque ya está diezmado)
ventana = 51
ecg_dec_smooth = sig.medfilt(ecg_dec, kernel_size=ventana)

# Interpolar de nuevo a la longitud original
x_dec = np.arange(len(ecg_dec))
x_full = np.linspace(0, len(ecg_dec)-1, len(ecg_one_lead))
baseline_multirate = np.interp(x_full, x_dec, ecg_dec_smooth)

# ECG corregido usando baseline multirate
ecg_corr_multirate = ecg_one_lead - baseline_multirate

# Gráfico
plt.figure(figsize=(12,5))
plt.plot(ecg_one_lead, label='ECG original', alpha=0.5)
plt.plot(baseline_multirate, label='Baseline multirate', linewidth=2)
plt.plot(ecg_corr_multirate, label='ECG corregido (multirate)', linewidth=1)
plt.title('BONUS: Estimación de línea de base usando multirate')
plt.xlabel('Muestras')
plt.legend()
plt.grid(True)
plt.show()
