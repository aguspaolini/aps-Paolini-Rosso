# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 19:41:53 2025

@author: Admin
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')

# plt.figure()
# plt.plot(ecg_one_lead)

#PSD
promedios = 20
nperseg = ecg_one_lead.shape[0] // promedios

print(nperseg)

nfft = 1*nperseg

f, Pxx = sig.welch(ecg_one_lead, fs=fs_ecg, window="hamming", nperseg=nperseg, nfft=nfft)

plt.figure()
plt.plot(f, Pxx)   #espectro pasa bajos
plt.xlim(0, 50)
plt.grid(True)
plt.show()

#Ancho de banda
df = f[1] - f[0]    #delta f
pot_total = np.sum(Pxx)* df
acumulada = np.cumsum(Pxx)*df

# índice del primer valor donde se alcanza el 95% de la potencia
indice = np.argmax(acumulada >= 0.95 * pot_total)
bw_ecg = f[indice]

print("Ancho de banda ECG:", bw_ecg, "Hz")
#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

# # Cargar el archivo CSV como un array de NumPy
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')

plt.figure()
plt.plot(ppg)

promedios = 30
nperseg = ecg_one_lead.shape[0] // promedios
nfft = 1*nperseg
print(nperseg)

f2, Pxx2 = sig.welch(ppg, fs=fs_ppg, window="hamming", nperseg=nperseg, nfft=nfft)


plt.figure()
plt.plot(f2, Pxx2)   
plt.xlim(0, 20)
plt.grid(True)
plt.show()
#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

plt.figure()
plt.plot(wav_data)

#si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
import sounddevice as sd
sd.play(wav_data, fs_audio)


