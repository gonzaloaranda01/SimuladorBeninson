

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
    VecEventTimeAmp=func_ModulateEvent(VecEventTime,VecEventAmp)
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




#--------------------------------------------------------------------------
# GENERACION DE PULSO TIPICO
#--------------------------------------------------------------------------
# Matlab [ Pulse , t_pulse ] = func_CreatePulse(cfg_Ts,cfg_Tau,cfg_Stages);
def func_CreatePulse(Ts,Tau,NumStages):
    """
    Modelar un pulso eléctrico. 
    La expresión representa la respuesta de un sistema de primer orden a un escalón. Aquí, t_pulse sería el tiempo, 
    Tau es la constante de tiempo del sistema, y NumStages es el número de etapas.
    Esta expresión implica una respuesta exponencial creciente seguida por un decaimiento exponencial. :
        - (tpulse/tau)**NumStages(representa un crecimiento exponencial.
        - exp(-tpulse/tau)representa el decaimiento exponencial.
    Multiplicar estas dos partes juntas implica que el pulso crece exponencialmente hasta alcanzar su máximo y luego decae exponencialmente.
    """
    if NumStages == 0:
        Tw = 7 * Tau
    elif NumStages == 1:
        Tmax = NumStages * Tau
        Tw = Tmax + 9 * Tmax
    elif NumStages == 2:
        Tmax = NumStages * Tau
        Tw = Tmax + 6 * Tmax
    else:
        Tmax = NumStages * Tau
        Tw = Tmax + 4 * Tmax

    t_pulse = np.arange(0, Tw + Ts, Ts)

    Pulse = ((t_pulse / Tau) ** NumStages) * np.exp(-t_pulse / Tau)
    Pulse = Pulse / np.max(Pulse)

    return Pulse, t_pulse
    

#--------------------------------------------------------------------------
# GENERACION DE TIEMPOS DE EVENTOS
#--------------------------------------------------------------------------
# [VecEventTime , t_abs] = func_CreateEventTime(cfg_Rin,cfg_Tmed,cfg_Ts,cfg_TimeRandomness);
def func_CreateEventTime(Rin, Tmed, Ts, RandomNotPeriodic):
    """
    Rin = Tasa configurada
    T med = Tiempo de medición
    Ts = Tiempo de muestreo
    RandomNotPeriodic = es la señal periodica o no. En este último caso se generan pulsos espaciados con distribución exponencial
    Genera un tren de deltas equiespaciadas o espaciadas por un dt de districuión exponencial
    """
    t_abs = np.arange(0, Tmed + Ts, Ts)
   
    # Si se configura la generacion de eventos aleatorias
    if RandomNotPeriodic == 1:
        NumEvents = round(Rin * Tmed)                     # Calculo el número de eventos total 
        lambda_val = Rin                                  # lamda es la tasa
        u = np.random.rand(NumEvents)                     # Genero números aleatorios con el método de la inversa
        VecDeltaT = (-1 / lambda_val) * np.log(1 - u)     # dt = (-1/lamda)*math.log(1-random.random()) otra manera
        time_OutOfGrid = np.cumsum(VecDeltaT)

        # Genero el vector de eventos detectados
        VecEvent, _ = np.histogram(time_OutOfGrid, bins=t_abs)
        # elimino los eventos apilados en el último instante de simulación
        VecEvent[-1] = 0
        VecEvent = np.concatenate((VecEvent,[0])) # le agrego un valor para que la dimensión sea igual a t
     # Si se configura de maneta periodica la generacion de eventos
    else:
        Dt = 1 / Rin
        Num = round(Dt / Ts)
        VecEvent = np.zeros_like(t_abs)                   # crea un arreglo VecEvent con la misma forma y tipo de datos que el arreglo t_abs, 
                                                          # pero todos los elementos son inicializados a cero.
        VecEvent[::Num] = 1                               # VecEvent(1:Num:end) = 1 ;
        VecEvent[-1] = 0
        #VecEvent = np.concatenate((VecEvent,[0])) # le agrego un valor para que la dimensión sea igual a t
    
    return VecEvent, t_abs

#--------------------------------------------------------------------------
# GENERACION DE AMPLITUDES
#--------------------------------------------------------------------------
# function [VecEventAmp ] = func_CreateEventAmp( NumEvents , f_x , x)
def func_CreateEventAmp(NumEvents, f_x, x):
    """
    --> Genera pulsos gaussianos, sino se puede ingresar por un archivo
    Amp_FotoPico = 5 * Volt;    % valor de tension correspondiente a la amplitud del pulso mas probable
    DetectorResolution = 3 ;    % resolucion del detector en porciento FWHM/Ecentral = 
    [ f_x , x ]=func_CreateAmpDistribution(AmpRandomness,DetectorResolution,Amp_FotoPico);
    
    """
    NBins = len(f_x)
    NumEvents = int(NumEvents)
    # Normalizo para que sean probabilidades y su integral sea "1"
    # Esto debería hacerse fuera de esta función, dependiendo del contexto
    Acumulada = np.cumsum(f_x)
    AcumuladaNormalizada = Acumulada / np.max(Acumulada)

    Uniforme = np.random.rand(int(NumEvents))
    MyOnes = np.ones(NBins)
    VecEventAmp = np.zeros(NumEvents)

    for i in range(NumEvents):
        RealizacionActual = Uniforme[i]
        VecFind = RealizacionActual * MyOnes
        # Encuentra el índice del valor mínimo
        IndexMin = np.argmax(AcumuladaNormalizada >= VecFind)
        VecEventAmp[i] = x[IndexMin]

    return VecEventAmp



#--------------------------------------------------------------------------
# GENERACION Distribución del pulso
#--------------------------------------------------------------------------
# Matlab function 
# [ f_x , x ]=func_CreateAmpDistribution(AmpRandomness,DetResolution,AmpPico)
def func_CreateAmpDistribution(AmpRandomness, DetResolution, AmpPico):
    if AmpRandomness == 0:
        Resolution = 0.0000001
    else:
        Resolution = DetResolution

    Media = AmpPico
    Desvio = (Resolution * Media) / (100 * 2.35)
    # Ajustar el rango de x según tus necesidades
    x = np.arange(0.1, 10.001, 0.001)
    # f_x = gauss2mf(x, [Desvio, Media, Desvio, Media])
    f_x = norm.pdf(x, Media, Desvio)

    return f_x, x


#--------------------------------------------------------------------------
# MODULACION DE AMPLITUDES DE EVENTOS
#--------------------------------------------------------------------------
#[VecEventTimeAmp]=func_ModulateEvent(VecEventTime,VecEventAmp);
def func_ModulateEvent(VecEventTime, VecAmp):
    VecEventTimeAmp = np.zeros_like(VecEventTime, dtype=float)
    Index = np.arange(1, len(VecEventTime) + 1)
    VecIndexEvent = Index[VecEventTime.astype(bool)]

    PosActualAmp = 0
    for i in VecIndexEvent:
        NumEvent = int(VecEventTime[i - 1])
        VecEventTimeAmp[i - 1] = np.sum(VecAmp[PosActualAmp : PosActualAmp + NumEvent])
        PosActualAmp = PosActualAmp + NumEvent

    return VecEventTimeAmp


def func_CreateAnalogSignal(Pulse, VecEvent):
    y_t = lfilter(Pulse, 1, VecEvent)
    return y_t


#--------------------------------------------------------------------------
# GENERACION SEÑAL DE RUIDO
#--------------------------------------------------------------------------
# [ n_t  ] = func_CreateNoiseSignal( Pulse , cfg_SNR , cfg_SNR_AmplitudeReference , N);
def func_CreateNoiseSignal(Pulse, SNR, Amp_Reference_SNR, N):
    """
    Elimino si tengo muchos niveles de pulso muy bajos
    Me quedo con la parte de pulso que tiene potencia,
    para que el cálculo de potencia mediante la varianza sea representativo
    """
    Signal_clean = Pulse[Pulse >= 0.001 * Pulse]
    Signal_clean_Amp = Amp_Reference_SNR * Signal_clean
    Pot_s = np.var(Signal_clean_Amp)
    Pot_n = Pot_s * 10**(-SNR / 10)
    n_t = np.sqrt(Pot_n) * np.random.randn(N)

    return n_t

#--------------------------------------------------------------------------
# SIMULACION DE SATURACION DE AMP FUERA DE RANGO LINEAL
#--------------------------------------------------------------------------
#[y_Amp] = func_Amp(y_analog,cfg_Amp_Gain,cfg_Amp_Vmax,cfg_Amp_Vmin);
def func_Amp(y_analog, Gain, Vmax, Vmin):
    y_s = Gain * y_analog
    y_s = np.clip(y_s, Vmin, Vmax)    #Acota y_s entre Vmin y Vmax
    return y_s


# --------------------------------------------------------------------------
# ADQUISICION DE SEÑAL (ADC)
# --------------------------------------------------------------------------
# [y_q] = func_SignalAdquired(y_Amp,cfg_ADC_Nbits,cfg_ADC_Vref);
def func_SignalAdquired(y_analog, Nbits, Vref):
    y_s = np.copy(y_analog)
    # Elimino la parte negativa
    GND = 0
    y_s[y_s < GND] = GND
    # Satura en 2^Nbits - 1
    y_s[y_s > Vref] = Vref
    y_q = np.round((y_s / Vref) * (2**Nbits - 1))

    return y_q


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