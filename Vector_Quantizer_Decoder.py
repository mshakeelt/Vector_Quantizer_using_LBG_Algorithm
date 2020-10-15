from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pyaudio
from functions import *
import pickle
import scipy.spatial.distance as sd
from training import *
from scipy.spatial import voronoi_plot_2d,Voronoi


rate, data = wav.read("Track48.wav")  
Signal = data[:,1]

f1 = pickle.load(open("codebook.bin","rb"))
f2 = pickle.load(open("coded_vq_signal.bin","rb"))
f3 = pickle.load(open("coded_uniform_q_signal.bin","rb"))
f4 = pickle.load(open("original_audio.bin","rb"))

def decodeVQ(y,index):
   
   x = np.zeros((len(index),2))
   for i in range(len(index)):
      x[i] = y[int(index[i])]
   return x


decoded = decodeVQ(f1,f2)

decoded = np.reshape(decoded,(1,-1))

decode = decoded[0,:]



q = (2**16)/(2**4)  #stepsize, 4 bit accuracy
rec1 = f3*q
error1 = Signal-rec1[:]
error2 = Signal-decode

print("signal size",max(rec1),min(rec1))
print("quantization error for mid-tread:",error1)
print("quantization error for vq",error2)

print('***Size of each data***')
print('size of original_audio.bin: {} MB'.format(os.stat('original_audio.bin').st_size / 10 ** 6))
print('size of coded_uniform_q_signal.bin: {} MB'.format(os.stat("coded_uniform_q_signal.bin").st_size / 10 ** 6))
print('size of coded_vq_signal.bin: {} MB'.format(os.stat("coded_vq_signal.bin").st_size / 10 ** 6))

""" 
l2, = plt.plot(Signal)
l1, = plt.plot(decode)

plt.legend(handles = [l1,l2,],labels = ['vq','orignal'])
plt.show()

l2, = plt.plot(Signal)
l3, = plt.plot(rec1[:])
plt.legend(handles = [l2,l3],labels = ['orignal','mid-tread'])
plt.show() """

l1, = plt.plot(decode)
l2, = plt.plot(Signal)
l3, = plt.plot(rec1[:])
#plt.legend(handles = [l2,l1,l3,],labels = ['Original Signal','vq','mid-tread'])
plt.show()


