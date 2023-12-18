

#--------------------------------------------------------------------------
#                                                                         %
#             COMISION NACIONAL DE ENERGIA ATOMICA (CNEA)                 %
#                  
#                                                                         %
#--------------------------------------------------------------------------
#                                                                         %
#  Autor: Gonzalo Damian Aranda                                     %
#  Fecha: 07/12/2023                                                      %
#                                                                         %
#--------------------------------------------------------------------------
#                                                                         %
#  Descripcion:               
#  Aqui se encuentran las funciones utilizadas en el simulador basado en  %
#  el simulador realizado en Matlab                                       %
#
#                               Versión 1.0.0
#--------------------------------------------------------------------------

# -------------------------------------------------------------------------
# LIBRERIAS
# -------------------------------------------------------------------------
import numpy as np    # Manejo con arrays
import scipy
from scipy.stats import norm
from scipy.signal import lfilter
from funciones_signal import *
#from funciones_signal import signal

# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# FUNCION PRINCIPAL DEL SIMULADOR
# -------------------------------------------------------------------------
def func_Simulator(StructConfiguration):
    cfg_TimeRandomness = StructConfiguration['cfg_TimeRandomness']
    cfg_Ts = StructConfiguration['cfg_Ts']
    cfg_Tau = StructConfiguration['cfg_Tau']
    cfg_Stages = StructConfiguration['cfg_Stages']
    cfg_Rin = StructConfiguration['cfg_Rin']
    cfg_Tmed = StructConfiguration['cfg_Tmed']
    f_x = StructConfiguration['f_x']
    x = StructConfiguration['x']
    cfg_SNR = StructConfiguration['cfg_SNR']
    cfg_SNR_AmplitudeReference = StructConfiguration['cfg_SNR_AmplitudeReference']
    cfg_Amp_Gain = StructConfiguration['cfg_Amp_Gain']
    cfg_Amp_Vmax = StructConfiguration['cfg_Amp_Vmax']
    cfg_Amp_Vmin = StructConfiguration['cfg_Amp_Vmin']
    cfg_ADC_Nbits = StructConfiguration['cfg_ADC_Nbits']
    cfg_ADC_Vref = StructConfiguration['cfg_ADC_Vref']

   
    # GENERACION DE PULSO TIPICO
    [ Pulse , t_pulse ] = func_CreatePulse(cfg_Ts,cfg_Tau,cfg_Stages)
    # GENERACION DE TIEMPOS DE EVENTOS  
    [ VecEventTime , t_abs ] = func_CreateEventTime(cfg_Rin,cfg_Tmed,cfg_Ts,cfg_TimeRandomness)
    # GENERACION DE AMPLITUDES
    NumEvents =  sum(VecEventTime)
    VecEventAmp = func_CreateEventAmp( NumEvents , f_x , x )
    # CALCULO EL HISTOGRAMA DE ENERGIA IDEAL
    IdealSpectrum, _ = np.histogram(VecEventAmp, bins=x)
    IdealSpectrum = np.concatenate([IdealSpectrum, [0.0]])
    #IdealSpectrum = np.concatenate(IdealSpectrum,[0.0])
    # MODULACION DE AMPLITUDES DE EVENTOS
    VecEventTimeAmp= func_ModulateEvent(VecEventTime,VecEventAmp)
    # GENERACION SEÑAL ANALOGICA LIMPIA
    y_t   = func_CreateAnalogSignal(Pulse,VecEventTimeAmp)
    # GENERACION SEÑAL DE RUIDO
    N = len(VecEventTimeAmp) 
    n_t   = func_CreateNoiseSignal( Pulse , cfg_SNR , cfg_SNR_AmplitudeReference , N)
    # CONTAMINO CON RUIDO LA SEÑAL ANALOGICA
    y_analog = y_t + n_t 
    # SIMULACION DE SATURACION DE AMP FUERA DE RANGO LINEAL
    y_Amp = func_Amp(y_analog,cfg_Amp_Gain,cfg_Amp_Vmax,cfg_Amp_Vmin)
    # ADQUISICION DE SEÑAL (ADC)
    y_q = func_SignalAdquired(y_Amp,cfg_ADC_Nbits,cfg_ADC_Vref)
    
    return Pulse, t_pulse, IdealSpectrum, t_abs, VecEventTime, VecEventAmp, \
        VecEventTimeAmp, y_t, n_t, y_analog, y_Amp, y_q





# ---------------------------------------------------------------------------------------------
# FUNCION PARA CONTAR PULSOS ------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def Comp_simple(signal, u):
    """Esta funcion es un simple comparador con un unico umbral"""
    N = len(signal)
    salidaComp = np.zeros(np.size(signal))
    for i in range(N):
        if signal[i]>=u:
            salidaComp[i]=1
        else:
            salidaComp[i]=0
    return salidaComp

# ---------------------------------------------------------------------------------------------
# FUNCION PARA CONTAR PULSOS COn Histeresis----------------------------------------------------
# ---------------------------------------------------------------------------------------------
def Comp_histeresis(signal,Ue,Us):
    """Esta función devuelve el numero de pulos con histeresis."""
    ON = 1
    OFF = 0
    ESTADO = OFF
    N = len(signal)
    salidaComp = np.zeros(np.size(signal))

    for i in range(N):
   
        if ESTADO == 'OFF':
            if signal[i] >= Ue:
                ESTADO = 'ON'
                salidaComp[i]=1        
            else:
                salidaComp[i]=0
                ESTADO = 'OFF'
        
        elif ESTADO == 'ON':
            if signal[i] < Us:
                salidaComp[i]=0
                ESTADO = 'OFF'
            else:
                ESTADO = 'ON'
                salidaComp[i]=1
                
        else:
            ESTADO = 'OFF'

    return salidaComp

# ---------------------------------------------------------------------------------------------
# FUNCION PARA CONTAR PULSOS ------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def PulseCountReload(signal,Ue,Us):
    """Esta función devuelve el numero de pulos con histeresis."""
    ON = 1
    OFF = 0
    ESTADO = OFF
    # DEAD TIME
    RT = 0
    LT = 0
    DT = 0
    N = len(signal)
    NumPulsos = 0

    for i in range(N):
        RT = RT+1
        if ESTADO == 'OFF':
            if signal[i] >= Ue:
                ESTADO = 'ON'
                NumPulsos += 1
                DT = DT + 1
            else:
                ESTADO = 'OFF'
        
        elif ESTADO == 'ON':
            if signal[i] < Us:
                ESTADO = 'OFF'
            else:
                DT = DT + 1
                ESTADO = 'ON'

        else:
            ESTADO = 'OFF'

    LT = RT - DT
    return NumPulsos, RT, DT, LT

# ---------------------------------------------------------------------------------------------
# FUNCION PARA CONTAR PULSOS ------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def PulseCount(signal,Ue,Us):
    """Esta función devuelve el numero de pulos."""
    ON = 1
    OFF = 0
    ESTADO = OFF
    N = len(signal)
    NumPulsos = 0

    for i in range(N):
        
        if ESTADO == 'OFF':
            if signal[i] >= Ue:
                ESTADO = 'ON'
                NumPulsos += 1
            else:
                ESTADO = 'OFF'
        
        elif ESTADO == 'ON':
            if signal[i] < Us:
                ESTADO = 'OFF'
            else:
                ESTADO = 'ON'

        else:
            ESTADO = 'OFF'

    return NumPulsos