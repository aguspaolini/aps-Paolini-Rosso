# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 16:57:36 2025

@author: aguspaolini
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#%% ======================== Generacion de señal senoidal ===================

def funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    
    t_muestra = 1/fs  # Obtengo tiempo al que se toma cada muestra  [s]
    
    tt = np.arange(nn) [:, np.newaxis] *t_muestra #(Nx1)
    xx = vmax*np.sin(2*np.pi*ff*tt + ph) + dc
    
    return tt, xx

def grafico(x, y, titulo, eje_x, eje_y):   #Función para graficar
    plt.plot(x, y, 'r', label = 'Señal')   # Utilizo linea fina
    plt.xlabel(eje_x)
    plt.ylabel(eje_y)
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
vmax = 3  # [V]
dc = 2 # Valor medio [V]
ff = 3  # Frecuencia [Hz]
ph = 0  # Fase en radianes
nn = 1000  # Cantidad de muestras
fs = 1000  # Frecuencia de muestreo 

tt, xx = funcion_sen(vmax, dc, ff, ph, nn, fs)
grafico(tt, xx, 'Señal senoidal', 'tiempo [s]', 'Amplitud [V]')


#%%================================ Bonus ======================================

t_1, x_1 = funcion_sen(vmax, 0, 500, np.pi, nn, fs)
grafico(t_1, x_1, 'Frecuencia = 500 Hz', 'tiempo [s]', 'Amplitud [V]')

t_2, x_2 = funcion_sen(vmax, 0, 999, ph, nn, fs)
grafico(t_2, x_2, 'Frecuencia = 999 Hz', 'tiempo [s]', 'Amplitud [V]')

t_3, x_3 = funcion_sen(vmax, 0, 1001, ph, nn, fs)
grafico(t_3, x_3, 'Frecuencia = 1001 Hz', 'tiempo [s]', 'Amplitud [V]')

t_4, x_4 = funcion_sen(vmax, 0, 2001, ph, nn, fs)
grafico(t_4, x_4, 'Frecuencia = 2001 Hz', 'tiempo [s]', 'Amplitud [V]')


def funcion_triangular(vmax=1, dc=0, ff=1, ancho=0.5, nn=1000, fs=1000):
    t_muestra = 1/fs  # Obtengo tiempo al que se toma cada muestra  [s]
    
    tt = np.arange(nn) [:, np.newaxis] *t_muestra #(Nx1)
    xx = vmax*signal.sawtooth(2*np.pi*ff*tt, ancho) + dc    # signal.sawtooth es señal diente de sierra, ajusto ancho
    
    return tt, xx

t_5, x_5 = funcion_triangular(vmax, 0, 5, 0.5, nn, fs)
grafico(t_5, x_5, 'Señal triangular','tiempo [s]', 'Amplitud [V]')