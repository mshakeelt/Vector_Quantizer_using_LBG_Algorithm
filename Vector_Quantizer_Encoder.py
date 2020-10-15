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


rate, data = wav.read("Track48.wav")   # signal
rate, data = wav.read("newaudio.wav")
Signal = data[:,1]
Signal = np.reshape(Signal,(-1,N))   # 1a
#print(Signal)
f1 = pickle.load(open("codebook.bin","rb"))
f2 = pickle.load(open("voronoi_regions.bin","rb"))

def encodeVQ(y,b,x):
   
   distance = np.zeros(M)  #Initiate Euclidean distance array
   row,col = x.shape
   index = np.zeros(row)

   for i in range(row):
      for j in range(M):
         distance[j] = sd.euclidean(y[j],x[i])  # distance between y and x
      index[i] = np.argmin(distance)   #find minimum distance for all x
      #print("distance ", i)
   return index


#index = encodeVQ(f1,f2,Signal)
#pickle.dump(index,open("coded_vq_signal.bin","wb"),1)

plot2D(f1,f2,Signal)

