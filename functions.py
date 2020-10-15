#Functions

from scipy import signal, stats, integrate
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import struct
import wave
from numpy import clip

opened=0
stream=[]

def sound(audio,  samplingRate):
  #funtion to play back an audio signal, in array "audio"
    import pyaudio
    if len(audio.shape)==2:
       channels=audio.shape[1]
       print("Stereo")
    else:
       channels=1
       print("Mono")
    p = pyaudio.PyAudio()
    # open audio stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplingRate,
                    output=True)
                    
    #Clipping to avoid overloading the sound device:
    audio=np.clip(audio,-2**15,2**15-1)
    sound = (audio.astype(np.int16).tostring())
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return  



def wavread(sndfile):
   "This function implements the wavread function of Octave or Matlab to read a wav sound file into a vector s and sampling rate info at its return, with: import sound; [s,rate]=sound.wavread('sound.wav'); or s,rate=sound.wavread('sound.wav');"

   import wave
   wf=wave.open(sndfile,'rb')
   nchan=wf.getnchannels()
   bytes=wf.getsampwidth()
   rate=wf.getframerate()
   length=wf.getnframes()
   print("Number of channels: ", nchan)
   print("Number of bytes per sample:", bytes)
   print("Sampling rate: ", rate)
   print("Number of samples:", length)
   length=length*nchan
   data=wf.readframes(length)
   if bytes==2: #2 bytes per sample:
      shorts = (struct.unpack( 'h' * length, data ))
   else:  #1 byte per sample:
      shorts = (struct.unpack( 'B' * length, data ))
   wf.close
   shorts=np.array(shorts)
   if nchan> 1:
      shorts=np.reshape(shorts,(-1,nchan))
   return shorts, rate


def wavwrite(snd,Fs,sndfile):
   """This function implements the wawwritefunction of Octave or Matlab to write a wav sound file from a vector snd with sampling rate Fs, with: 
import sound; 
sound.wavwrite(snd,Fs,'sound.wav');"""

   import wave
   import pylab
 
   WIDTH = 2 #2 bytes per sample
   #FORMAT = pyaudio.paInt16
   CHANNELS = 1
   #RATE = 22050
   RATE = Fs #32000

   length=pylab.size(snd)
   
   wf = wave.open(sndfile, 'wb')
   wf.setnchannels(CHANNELS)
   wf.setsampwidth(WIDTH)
   wf.setframerate(RATE)
   data=struct.pack( 'h' * length, *snd )
   wf.writeframes(data)
   wf.close()



def record(time, Fs):
   "Records sound from a microphone to a vector s, for instance for 5 seconds and with sampling rate of 32000 samples/sec: import sound; s=sound.record(5,32000);"
   
   import numpy
   global opened
   global stream
   CHUNK = 1000 #Blocksize
   WIDTH = 2 #2 bytes per sample
   CHANNELS = 1 #2
   RATE = Fs  #Sampling Rate in Hz
   RECORD_SECONDS = time

   p = pyaudio.PyAudio()

   a = p.get_device_count()
   print("device count=",a)
   
   #if (opened==0):
   if(1):
     for i in range(0, a):
        print("i = ",i)
        b = p.get_device_info_by_index(i)['maxInputChannels']
        print("max Input Channels=", b)
        b = p.get_device_info_by_index(i)['defaultSampleRate']
        print("default Sample Rate=", b)

     stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                #input_device_index=3,
                frames_per_buffer=CHUNK)
     opened=1           

   print("* recording")
   snd=[]
#Loop for the blocks:
   for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      #Reading from audio input stream into data with block length "CHUNK":
      data = stream.read(CHUNK)
      #Convert from stream of bytes to a list of short integers (2 bytes here) in "samples":
      #shorts = (struct.unpack( "128h", data ))
      shorts = (struct.unpack( 'h' * CHUNK, data ))
      #samples=list(shorts);
      samples=shorts
      #samples = stream.read(CHUNK).astype(np.int16)
      snd=numpy.append(snd,samples)
   return snd


def db_range(input_signal, dB): #To Play with new SNR
    c = 10.0**(dB/20.0)
    return input_signal/c

def signal_play(data, fs, channels):    #To play the audio
    p = pyaudio.PyAudio() #Initializing PYAudio
    if data.dtype == np.int8: #For 8 bit input
            #Opening the media stream
        stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=fs,
                    output=True)
        data = data.astype(np.int8).tostring()

    #Playing the media stream 
        stream.write(data)
        stream.close()
    else:
        stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=fs,
                    output=True)

        data = data.astype(np.int16).tostring()
    
        stream.write(data)
        stream.close() #Closing the media stream
   

def channels(filepath):
    data=wave.open(filepath) #Opeining WaveFile and acquiring data
    channels=wave.Wave_read.getnchannels(data) #Getting Channels
    return channels



def mid_tread(Signal_Data, bit_size):
    step = (float(np.amax(Signal_Data))-float(np.amin(Signal_Data))) / pow (2,bit_size)
    
    #In Mid Tread Quantizer index = round of Signal_Data/step
    index = np.round(Signal_Data/step)
    
    #Reconstruction of Signal
    #reconstruct = np.array(Signal_Data.shape)

    reconstruct=index*step
    #reconstruct= reconstruct.astype(np.int8)    
    return reconstruct
    
"""
Defining a function for Mid Rise Quantizer which takes input of Signal_Data
and Num of bit_size 
"""


def mid_rise(Signal_Data, bit_size):
  
    #step delta = Amax-Amin/2^N
    step = (float(np.amax(Signal_Data))-float(np.amin(Signal_Data))) / pow (2,bit_size)
    
    #In Mid Rise Quantizer index = floor of Signal_Data/step
    index = np.floor(Signal_Data/step)

    #Reconstruction of Signal
    #reconstruct = np.array(Signal_Data.shape)
    
    reconstruct=index * step + step/2
    #reconstruct= reconstruct.astype(np.int8)
    return reconstruct

def u_Law(Signal_Data, bit_size, quantizer):
    S_Max  =  float(np.amax(Signal_Data))
#    S_Min  = float(np.amin(Signal_Data))
    u = 255.0

    #u-Law Compression Expression
    Signal_y=np.sign(Signal_Data)*(np.log(1+ u* np.abs(Signal_Data/S_Max)))/np.log(1 + u)
    #Quantizer Selection

    if quantizer == 'midtread':
        Signal_yrek = mid_tread(Signal_y, bit_size)
        print("Signal has been uniformly quantized using Mid Tread Quantizer")
    elif quantizer == 'midrise':
        Signal_yrek = mid_rise(Signal_y, bit_size)
        print("Signal has been uniformly quantized using Mid Rise Quantizer")
    elif quantizer == 0:
        Signal_yrek=Signal_y
        print("Signal has not been uniformly quantized (Y=Yrek)")
            

   #u-law Expansion Expression
    reconstruct = np.sign(Signal_yrek)*(256**(np.abs(Signal_yrek))-1)*S_Max/u
    

    #reconstruct= reconstruct.astype(np.int8)
    return reconstruct
        

    

def SNR(Signal_Data, bit_size, quantizer):

#Checking Quantizer
    Eng_Signal=0.0
    Eng_Error=0.0
    if quantizer == 'midtread':
        Signal_Quantization = mid_tread(Signal_Data, bit_size)
    elif quantizer == "midrise":
        Signal_Quantization = mid_rise(Signal_Data, bit_size)
    elif quantizer == "ulaw2":
        Signal_Quantization=u_Law(Signal_Data, bit_size,'midtread')
    elif quantizer == "ulaw1":
        Signal_Quantization=u_Law(Signal_Data, bit_size,'midrise')   
#Error Signal    
    Error_Signal= Signal_Data - Signal_Quantization
#Energy in Original Signal
    Eng_Signal = np.sum(np.square(Signal_Data))
#Energy in Error Signal
    Eng_Error = np.sum(np.square(Error_Signal))
#SNR = 10* log10(Signal Energy/Quantization Error Energy
    SNR= 10 * np.log10(Eng_Signal/Eng_Error)
    return SNR

def Signal_gen(SamplingFreq, Freq, Amplitude, duration, wave):
    n=np.arange(-duration*SamplingFreq,duration*SamplingFreq,0.25).astype(np.float32)
#if phase in sawthooth is 1 or 0 its tilted right if or tilted left
#phase value 0.5 keeps it centered to give us trianglular waveform  

    if wave == 'triangle':
        return Amplitude * signal.sawtooth(2 * np.pi * (Freq/SamplingFreq) * n, 0.5)
    elif wave == 'sine':
        return Amplitude * np.sin(2 * np.pi * (Freq/SamplingFreq) * n).astype(np.float32)
