# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 10:29:10 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==================================================
# Defino los coeficientes de cada filtro
# ==================================================
# y(n) = b0*x(n) + b1*x(n-1) + ...
#       → H(z) = b0 + b1*z^-1 + ...

b_a = [1, 1, 1, 1]           # (a)
b_b = [1, 1, 1, 1, 1]        # (b)
b_c = [1, -1]                # (c)
b_d = [1, 0, -1]             # (d)

# Denominador (sistemas FIR → a = [1])
a = [1]

# ==================================================
# Cálculo de la respuesta en frecuencia
# ==================================================
w_a, h_a = signal.freqz(b_a, a)
w_b, h_b = signal.freqz(b_b, a)
w_c, h_c = signal.freqz(b_c, a)
w_d, h_d = signal.freqz(b_d, a)


# ==================================================
# Respuestas en frecuencia obtenidas analíticamente
# ==================================================
eps = 1e-12  # evita divisiones por cero

# a)
h_a_ana = np.exp(-1j*3*w_a/2) * (np.sin(2*w_a)/(np.sin(w_a/2) + eps))

# b)
h_b_ana = np.exp(-1j*2*w_b) * (np.sin(5*w_b/2)/(np.sin(w_b/2) + eps))

# c)
h_c_ana = np.exp(-1j*w_c/2) * (2j*np.sin(w_c/2))

# d)
h_d_ana = np.exp(-1j*w_d)*(2j*np.sin(w_d))

# ==================================================
# Grafico el módulo y la fase
# ==================================================
plt.figure(figsize=(12, 10))

# ----------- MÓDULO -----------
plt.subplot(2, 1, 1)
plt.plot(w_a, np.abs(h_a), color = 'm', label='Filtro (a): numérico')
plt.plot(w_a, np.abs(h_a_ana), color = 'cyan', linestyle= '--', label='Filtro (a): analítico')
plt.title('Respuesta en frecuencia - Módulo')
plt.xlabel('Frecuencia [rad/muestra]')
plt.ylabel('|H(e^{jω})|')
plt.grid(True)
plt.legend()

# ----------- FASE -----------
plt.subplot(2, 1, 2)
plt.plot(w_a, np.unwrap(np.angle(h_a)), color = 'g', label='Filtro (a): numérico')
plt.plot(w_a, np.unwrap(np.angle(h_a_ana)), color = 'r', linestyle= '--', label='Filtro (a): analítico')
plt.title('Respuesta en frecuencia - Fase')
plt.xlabel('Frecuencia [rad/muestra]')
plt.ylabel('Fase [rad]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 10))

# ----------- MÓDULO -----------
plt.subplot(2, 1, 1)
plt.plot(w_b, np.abs(h_b), color = 'm', label='Filtro (b): numérico')
plt.plot(w_b, np.abs(h_b_ana), color = 'cyan', linestyle= '--', label='Filtro (b): analítico')
plt.title('Respuesta en frecuencia - Módulo')
plt.xlabel('Frecuencia [rad/muestra]')
plt.ylabel('|H(e^{jω})|')
plt.grid(True)
plt.legend()

# ----------- FASE -----------
plt.subplot(2, 1, 2)
plt.plot(w_b, np.unwrap(np.angle(h_b)), color = 'g', label='Filtro (b): numérico')
plt.plot(w_b, np.unwrap(np.angle(h_b_ana)), color = 'r', linestyle= '--', label='Filtro (b): analítico')
plt.title('Respuesta en frecuencia - Fase')
plt.xlabel('Frecuencia [rad/muestra]')
plt.ylabel('Fase [rad]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 10))

# ----------- MÓDULO -----------
plt.subplot(2, 1, 1)
plt.plot(w_c, np.abs(h_c), color = 'm', label='Filtro (c): numérico')
plt.plot(w_c, np.abs(h_c_ana), color = 'cyan', linestyle= '--', label='Filtro (c): analítico')
plt.title('Respuesta en frecuencia - Módulo')
plt.xlabel('Frecuencia [rad/muestra]')
plt.ylabel('|H(e^{jω})|')
plt.grid(True)
plt.legend()

# ----------- FASE -----------
plt.subplot(2, 1, 2)
plt.plot(w_c, np.unwrap(np.angle(h_c)), color = 'g', label='Filtro (c): numérico')
plt.plot(w_c, np.unwrap(np.angle(h_c_ana)), color = 'r', linestyle= '--', label='Filtro (c): analítico')
plt.title('Respuesta en frecuencia - Fase')
plt.xlabel('Frecuencia [rad/muestra]')
plt.ylabel('Fase [rad]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 10))

# ----------- MÓDULO -----------
plt.subplot(2, 1, 1)
plt.plot(w_d, np.abs(h_d), color = 'm', label='Filtro (d): numérico')
plt.plot(w_d, np.abs(h_d_ana), color = 'cyan', linestyle= '--', label='Filtro (d): analítico')
plt.title('Respuesta en frecuencia - Módulo')
plt.xlabel('Frecuencia [rad/muestra]')
plt.ylabel('|H(e^{jω})|')
plt.grid(True)
plt.legend()

# ----------- FASE -----------
plt.subplot(2, 1, 2)
plt.plot(w_d, np.unwrap(np.angle(h_d)), color = 'g', label='Filtro (d): numérico')
plt.plot(w_d, np.unwrap(np.angle(h_d_ana)), color = 'r', linestyle= '--', label='Filtro (d): analítico')
plt.title('Respuesta en frecuencia - Fase')
plt.xlabel('Frecuencia [rad/muestra]')
plt.ylabel('Fase [rad]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()




# plt.figure(figsize=(12, 10))

# # ----------- MÓDULO -----------
# plt.subplot(2, 1, 1)
# plt.plot(w, 20*np.log10(np.abs(h_a)), label='Filtro (a): Media móvil 4')
# plt.plot(w, 20*np.log10(np.abs(h_b)), label='Filtro (b): Media móvil 5')
# plt.plot(w, 20*np.log10(np.abs(h_c)), label='Filtro (c): Diferenciador simple')
# plt.plot(w, 20*np.log10(np.abs(h_d)), label='Filtro (d): Diferenciador retardado')
# plt.title('Respuesta en frecuencia - Módulo (dB)')
# plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
# plt.ylabel('|H(e^{jω})| [dB]')
# plt.grid(True)
# plt.legend()

# # ----------- FASE -----------
# plt.subplot(2, 1, 2)
# plt.plot(w, np.unwrap(np.angle(h_a)), label='Filtro (a)')
# plt.plot(w, np.unwrap(np.angle(h_b)), label='Filtro (b)')
# plt.plot(w, np.unwrap(np.angle(h_c)), label='Filtro (c)')
# plt.plot(w, np.unwrap(np.angle(h_d)), label='Filtro (d)')
# plt.title('Respuesta en frecuencia - Fase')
# plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
# plt.ylabel('Fase [rad]')
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()