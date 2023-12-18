import numpy as np
import matplotlib.pyplot as plt
import funciones.FuncionesSimulador as func
from funciones.funciones_signal import func_CreateAmpDistribution
import matplotlib as plt
import matplotlib.pyplot as plt
import time
import yaml

#--------------------------------------------------------------------------
#                                                                         %
#             COMISION NACIONAL DE ENERGIA ATOMICA (CNEA)                 %
#                                                                         %
#--------------------------------------------------------------------------
#                                                                         %
#  Autor: Gonzalo Aranda                                                  %
#  Fecha: 11/12/2023                                                      %
#                                                                         %
#--------------------------------------------------------------------------
#                                                                         %
#  Descripcion:                                                           %
#  Simulador Multicanal Version 3.0  
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
# Inicializacion del entorno
#--------------------------------------------------------------------------


# En caso de tener que agregar directorios hacerlo aqui
#Path_to_SimulatorFunctions = 'functionsSimulator'
#sys.path.append(Path_to_SimulatorFunctions)

#Path_to_MyFunctions = 'functionTP' 
#sys.path.append(Path_to_MyFunctions)



#--------------------------------------------------------------------------
# Unidades
#--------------------------------------------------------------------------
Hz = 1 
khz = 1e3 * Hz  
Mhz = 1e3 * khz 
Ghz = 1e3 * Mhz 

sec = 1 / Hz  
ms  = 1 / khz 
us  = 1 / Mhz 
ns  = 1 / Ghz 

Volt = 1
dB = 1 

cps = 1 
kcps = 1e3 * cps 
Mcps = 1e3 * kcps 

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

# Tiempo
PERIODIC_TIME = 0 
RANDOM_TIME = 1 

# Amplitud
CONSTANT_AMP = 0 
RANDOM_AMP = 1 

YES = 1 
NOT = 0 

# Declaración de parámetros de configuración y experimento
FileNameStructConfiguration = 'stConfig.yaml'
StructConfiguration = {}

# Parámetros de configuración
StructConfiguration['cfg_TimeRandomness'] = PERIODIC_TIME
#StructConfiguration['cfg_TimeRandomness'] = 'RANDOM_TIME'

# Aleatorio en amplitud o constante
# Si es aleatorio, tengo que darle una PDF
#StructConfiguration['cfg_AmpRandomness'] = CONSTANT_AMP
StructConfiguration['cfg_AmpRandomness'] = (RANDOM_AMP)

# StructConfiguration['RandomRepetitive'] = NOT
StructConfiguration['cfg_RandomRepetitive'] = YES             # Pongo una semilla

# Parámetros del experimento
StructConfiguration['cfg_Version'] = '3.0'

# Resolución temporal de la simulación / frecuencia de muestreo
StructConfiguration['cfg_Ts'] = 1 * us

# Tasa de detección: lo que llega al detector
StructConfiguration['cfg_Rin'] = 1 * kcps

# Lapso de tiempo absoluto simulado
StructConfiguration['cfg_Tmed'] = 100 * ms

# Constante de tiempo del amplificador - duración pulsos analógicos
StructConfiguration['cfg_Tau'] = 5 * us

# Número de etapas integradoras en un Amplificador CR-(RC)^N con todas las
# constantes de tiempo iguales
StructConfiguration['cfg_Stages'] = 0  # 0 exponencial, 1 tiende a pseudoexponencial

# Nivel de ruido
# StructConfiguration['cfg_SNR'] = inf * dB
StructConfiguration['cfg_SNR'] = 50 * dB

# El SNR está calculado contra esta amplitud
StructConfiguration['cfg_SNR_AmplitudeReference'] = 5 * Volt

# Niveles de saturación del Amplificador
StructConfiguration['cfg_Amp_Vmax'] = 10 * Volt
StructConfiguration['cfg_Amp_Vmin'] = -10 * Volt
StructConfiguration['cfg_Amp_Gain'] = 1

# ADC
StructConfiguration['cfg_ADC_Nbits'] = 8
StructConfiguration['cfg_ADC_Vref'] = 10 * Volt


print('***************************************************************')
print('----- Inicialización de Generación de Números Aleatorios ------')

if StructConfiguration['cfg_RandomRepetitive'] == YES:
    StructConfiguration['Semilla_EventTime'] = 0
else:
    # En Python, puedes usar la función time para obtener la hora actual
    StructConfiguration['Semilla_EventTime'] = int(sum([100 * x for x in time.localtime()]))

# No existe una función directa en Python equivalente a func_NewRandomSeed,
# puedes utilizar np.random.seed para configurar la semilla del generador de números aleatorios de NumPy.

print('Semilla_EventTime:', StructConfiguration['Semilla_EventTime'])
print('*************************************************************\n')
print('*************************************************************')
print('------------ Generación de Espectro de Energía --------------')

# Carga del espectro desde un archivo MAT (supongo que tienes un archivo Espectro_1024.mat)
# Si no, puedes comentar la línea y usar tus propios datos para f_x y x
# Espectro_1024.mat debe contener las variables 'Espectro' y 'Vec_Channels'
# spectrums_path = 'spectrums\\Espectro_1024.mat'
# data = scipy.io.loadmat(spectrums_path)
# f_x = data['Espectro'].flatten()
# Vec_Channels = data['Vec_Channels'].flatten()


# Generación de pulsos gaussianos
# genera pulsos gaussianos, sino se puede ingresar por un archivo
StructConfiguration['cfg_Amp_FotoPico'] = 5 * Volt      # valor de tension correspondiente a la amplitud del pulso mas probable
StructConfiguration['cfg_DetectorResolution'] = 3            # resolucion del detector en porciento FWHM/Ecentral = 


# -------------------------------------------------------------------------------------------------------------------------
# CARGO LAS VARIABLES
TimeRandomness = StructConfiguration['cfg_TimeRandomness']
AmpRandomness = StructConfiguration['cfg_AmpRandomness'] 
RandomRepetitive = StructConfiguration['cfg_RandomRepetitive'] 
Version = StructConfiguration['cfg_Version']
Ts = StructConfiguration['cfg_Ts']
Rin = StructConfiguration['cfg_Rin'] 
Tmed = StructConfiguration['cfg_Tmed']

Amp_FotoPico = StructConfiguration['cfg_Amp_FotoPico'] 
DetectorResolution = StructConfiguration['cfg_DetectorResolution']

[ f_x , x ]= func_CreateAmpDistribution(AmpRandomness,DetectorResolution,Amp_FotoPico)

# Visualización del espectro de entrada

fig1 = plt.figure('Espectro de Entrada')
plt.plot(x, f_x,'m',lw = '0.5')
plt.title('MCA Entrada')
plt.xlabel('Canales')
plt.ylabel('Cuentas')
plt.grid()
plt.show()

# Configuración de la estructura StructConfiguration
StructConfiguration['f_x'] = f_x  # probabilidad
StructConfiguration['x'] = x

# Guardar StructConfiguration en un archivo (suponiendo que FileNameStructConfiguration está definido)
#with open(FileNameStructConfiguration, 'w') as archivo_yaml:
    #yaml.dump(StructConfiguration, archivo_yaml, default_flow_style=False)


print('*************************************************************')
print('*************************************************************')
print('--------------------- Inicio Simulador ----------------------')

# Medir el tiempo de ejecución
start_time = time.time()
# Llamada a la función func_Simulator
resultados = func.func_Simulator(StructConfiguration)
# Calcular el tiempo transcurrido
time_lapse = time.time() - start_time
# Desempaquetar los resultados si es necesario
Pulse, t_pulse, IdealSpectrum, t_abs, VecEventTime, VecEventAmp, VecEventTimeAmp, y_t, n_t, y_analog, y_Amp, y_q = resultados
# Medir el tiempo de simulación
start_time = time.time()

# Imprimir el tiempo de simulación
print(f'Tiempo de Simulación de señal: {time_lapse} [seg]')
print('*************************************************************\n')

# Gráficos y resultados de simulación

# Pulse
fig2 = plt.figure('Shaping Amplifier')
plt.plot(t_pulse/us, Pulse,'r',lw = 0.5)
titulo = f'h(t) Shaping Amplfier CR-(RC)^{StructConfiguration["cfg_Stages"]}'
plt.title(titulo)
plt.xlabel('time [us]')
plt.ylabel('Amplitud normalizada')
plt.grid()
plt.show()

#--------------
# Espectro Ideal
#--------------
fig3 = plt.figure('Espectro de Amplitudes Ideal')
plt.plot(x, IdealSpectrum)
plt.title('Espectro de Amplitudes Ideal')
plt.xlabel('canales')
plt.grid()
plt.show()


# Gráficos y resultados de simulación
# Signals
FS= 6
fig, axs = plt.subplots(6, 1, sharex=True, figsize=(8, 10))

# Subplot 1: Event Time
axs[0].stem(t_abs/us, VecEventTime, linefmt='.-', basefmt=" ", markerfmt='.')
axs[0].set_ylabel('Event Time',fontsize=FS)
axs[0].set_title('Signals Out')

# Subplot 2: Event Time Amp
axs[1].stem(t_abs/us, VecEventTimeAmp, linefmt='.-', basefmt=" ", markerfmt='.')
axs[1].set_ylabel('Event Time Amp',fontsize=FS)

# Subplot 3: Signal Detector
axs[2].plot(t_abs/us, y_t,'b',lw = 0.5)
axs[2].set_ylabel('Signal Detector',fontsize=FS)

# Subplot 4: Signal with Noise
axs[3].plot(t_abs/us, y_analog,'b',lw = 0.5)
axs[3].set_ylabel('Signal with Noise',fontsize=FS)

# Subplot 5: Signal Amplified
axs[4].plot(t_abs/us, y_Amp,'b',lw = 0.5)
axs[4].set_ylabel('Signal Amplified',fontsize=FS)
axs[4].set_ylim([StructConfiguration["cfg_Amp_Vmin"], 1.1 * StructConfiguration["cfg_Amp_Vmax"]])

# Subplot 6: Signal Acquired
#axs[5].stairs(np.concatenate[t_abs/us,[0.0]] , y_q)
#y_q = np.concatenate([y_q, [0.0]])
#axs[5].plot(t_abs/us , y_q)
axs[5].stairs(y_q)
axs[5].set_ylabel('Signal Acquired', fontsize=FS)
axs[5].set_xlabel('time [us]')

# Ajustar diseño y mostrar gráfico
plt.grid()
plt.tight_layout()
plt.show()
