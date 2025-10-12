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
promedios = 30
nperseg = ecg_one_lead.shape[0] // promedios

print(nperseg)

nfft = 5*nperseg

f, Pxx = sig.welch(ecg_one_lead, fs=fs_ecg, window="hamming", nperseg=nperseg, nfft=nfft)

#Ancho de banda
df = f[1] - f[0]    #delta f
pot_total = np.sum(Pxx)* df
acumulada = np.cumsum(Pxx)*df

# índice del primer valor donde se alcanza el 99% de la potencia
indice = np.where(acumulada >= 0.99 * pot_total)[0][0]
bw_ecg = f[indice]

print("Ancho de banda ECG:", bw_ecg, "Hz")


plt.figure()
plt.plot(f, Pxx)   #espectro pasa bajos
plt.xlabel('Frecuencia [Hz]')
plt.axvline(bw_ecg, color='r', linestyle='--')
plt.xlim(0, 50)
plt.title('ECG')
plt.grid(True)
plt.show()

# plt.figure()
# plt.plot(f, acumulada / pot_total)
# plt.axvline(bw_ecg)
# plt.xlabel('Frecuencia [Hz]')
# plt.xlim(0,40)
# plt.ylabel('Potencia acumulada [%]')
# plt.grid(True)
# plt.show()

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

# plt.figure()
# plt.plot(ppg)

promedios2 = 45
nperseg2 = ppg.shape[0] // promedios2
nfft2 = 2*nperseg2
print(nperseg2)

f2, Pxx2 = sig.welch(ppg, fs=fs_ppg, window="hamming", nperseg=nperseg2, nfft=nfft2)

#Ancho de banda
df2 = f2[1] - f2[0]    #delta f
pot_total2 = np.sum(Pxx2)* df2
acumulada2 = np.cumsum(Pxx2)*df2

# índice del primer valor donde se alcanza el 99% de la potencia
indice2 = np.where(acumulada2 >= 0.99 * pot_total2)[0][0]
bw_ppg = f2[indice2]

print("Ancho de banda PPG:", bw_ppg, "Hz")

plt.figure()
plt.plot(f2, Pxx2) 
plt.xlabel('Frecuencia [Hz]')
plt.axvline(bw_ppg, color='g', linestyle='--')  
plt.xlim(0, 20)
plt.title('PPG')
plt.grid(True)
plt.show()


# plt.figure()
# plt.plot(f2, acumulada2 / pot_total2)
# plt.xlabel('Frecuencia [Hz]')
# plt.xlim(0,40)
# plt.ylabel('Potencia acumulada [%]')
# plt.grid(True)
# plt.show()
#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# plt.figure()
# plt.plot(wav_data)

promedios3 = 135
nperseg3 = wav_data.shape[0] // promedios3
nfft3 = 4*nperseg3
print(nperseg3)

f3, Pxx3 = sig.welch(wav_data, fs=fs_audio, window="hamming", nperseg=nperseg3, nfft=nfft3)

#Ancho de banda
df3 = f3[1] - f3[0]    #delta f
pot_total3 = np.sum(Pxx3)* df3
acumulada3 = np.cumsum(Pxx3)*df3

# índice del primer valor donde se alcanza el 99% de la potencia
indice3 = np.where(acumulada3 >= 0.99 * pot_total3)[0][0]
bw_audio_max = f3[indice3]

#índice del primer valor donde se el alcanza el 1% de la potencia
indice4 = np.where(acumulada3 >= 0.01 * pot_total3)[0][0]
bw_audio_min = f3[indice4]

print("Ancho de banda mínimo audio:", bw_audio_min, "Hz")
print("Ancho de banda máximo audio:", bw_audio_max, "Hz")

plt.figure()
plt.plot(f3, Pxx3) 
plt.xlabel('Frecuencia [Hz]')
plt.axvline(bw_audio_max, color='m', linestyle='--') 
plt.axvline(bw_audio_min, color='m', linestyle='--') 
plt.xlim(500, 3000)
plt.title('Audio')
plt.grid(True)
plt.show()


# plt.figure()
# plt.plot(f3, acumulada3 / pot_total3)
# plt.xlabel('Frecuencia [Hz]')
# #plt.xlim(0,40)
# plt.ylabel('Potencia acumulada [%]')
# plt.grid(True)
# plt.show()

#si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
#import sounddevice as sd
#sd.play(wav_data, fs_audio)


