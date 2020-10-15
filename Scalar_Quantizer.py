from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import functions as fn
import pickle

bitsize=4

#1a
Fs1, Signal = wavfile.read('Track48.wav','r')
Signal1 = Signal[:, 1]
pickle.dump(Signal1,open("original_audio.bin","wb"),1)
print(Fs1)



Sig_Duration= len(Signal)/Fs1
Time=np.linspace(0, Sig_Duration, len(Signal))
channels= fn.channels('Track48.wav')
#print("Playing Full Range Audio")
#fn.signal_play(Signal1,Fs1,channels)

#1b, Second file of 3 seconds
Fs2, Signal2 = wavfile.read("newaudio.wav",'r')
print(Fs2)
#2a
signal_reconstruct=fn.mid_tread(Signal1,bitsize)

#2c
step = (float(np.amax(Signal1))-float(np.amin(Signal1))) / pow (2,bitsize)
index = np.round(Signal1/step)
pickle.dump(index,open("coded_uniform_q_signal.bin","wb"),1)  #save 1a

#2b
figure,(w1) = plt.subplots(1)
#Plotting Full Range Audio
w1.plot(Time,Signal1,'r', label='Original Signal')
w1.plot(Time,signal_reconstruct,'b', label='Reconstructed Signal')
w1.set_ylabel('Amplitude(A)') #Y-axis
w1.set_xlabel('Time(T)                                                                                                                                  ') #X-axis
w1.set_title('Original vs Quantized Signal')
leg=w1.legend()
plt.show()




