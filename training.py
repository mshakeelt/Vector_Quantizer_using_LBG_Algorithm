from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pyaudio
from functions import *
import pickle
import scipy.spatial.distance as sd
import os.path
from scipy.spatial import voronoi_plot_2d,Voronoi

N = 2
M = 256   # number of codevectors

def readAudio1b():

   rate, data = wav.read("newaudio.wav")
   #print(data.shape)
   return data[:,1]

def readAudio1a():

   rate, data = wav.read("Track48.wav")
   #print(data.shape)
   return data[:,1]

def step2(y):
   print("step2")
   #step two

   num = len(y)
   boundarynum = (num+1)*num/2       # number of boundaries
   boundary = np.zeros((int(boundarynum),2))  #Initiate 

   codeVec = np.zeros((2,2))   # calculate the b between two codevector 
   b_seq = 0   # the sequence number of b

   for i in range(0,num-1):
       codeVec[0] = y[i,:]
       for j in range(i+1,num):
          codeVec[1] = y[j,:]
          boundary[b_seq] = (codeVec[0] + codeVec[1])/2
          b_seq = b_seq+1

   return boundary 

def step3(trainSeq,y):
   #print("step3")
   #step three
   distance = np.zeros(M)  #Initiate Euclidean distance array(for one sample)
   row,col = trainSeq.shape
   index = np.zeros(row)   # x belongs to which codevector

   print("compute distance")
   for i in range(row):
      for j in range(M):
         distance[j] = sd.euclidean(y[j],trainSeq[i])
      index[i] = np.argmin(distance)   #find minimum distance for x
      #print("distance",i)

   vector = np.zeros((M,2))   # the sum of training sequences for regions 
   vectornum = np.zeros(M)   # how many samples in each region
   yn = np.zeros(y.shape)   # new Y
 
   print("compute new y")
   for i in range(row):
      for j in range(M):
         if index[i] == j :   # if x[i] is in the region j
            vector[j] = vector[j] + trainSeq[i]  # add x[i] to region j
            vectornum[j] = vectornum[j] + 1   # the number of x in j
   
   for i in range(M):
      if vectornum[i] != 0:
         yn[i] = vector[i]/vectornum[i]    # new Y

   difference = np.sum(np.abs(yn-y))   # the difference between y and new y

   return yn,difference
 
def plot2D(y,boundary,trainSeq):
    
    vor = Voronoi(y)

    fig = voronoi_plot_2d(vor,show_points = None,show_vertices = None,line_colors = 'g')

    plt.scatter(trainSeq[:,0],trainSeq[:,1], color = 'b', s=10)  
    plt.scatter(y[:,0],y[:,1],color = 'r', marker = '*',s=30)
    plt.show()

    return

if __name__ == '__main__':

   voice = readAudio1b()
   voice2 = readAudio1a()
   
   #step one
   Amplitude = np.max(voice) - np.min(voice) #Amplitude of training set
   y = np.zeros((M,2))    #Initiate codebook
   

   trainSeq = np.reshape(voice,(-1,N))  #Initiate Xk

   Seq = np.linspace(np.min(voice2),np.max(voice2),M)
   y[:,0] = Seq    #Assigned codebook vectors
   y[:,1] = Seq

   difference = 10000   #The difference between y in two cycle
   cycle = 0
 
   while difference > 5000 :   #cycle for step three
     y,difference = step3(trainSeq,y)
     cycle = cycle +1
     print("cycle is",cycle,"difference is",difference)
 
   boundary = step2(y)  

   print("The codebook is",y)

   pickle.dump(y,open("codebook.bin","wb"),1)
   pickle.dump(y,open("voronoi_regions.bin","wb"),1)
   #f1 = pickle.load(open("codebook.bin","rb"))
   #f1 = pickle.load(open("voronoi_regions.bin","rb"))


   plot2D(y,boundary,trainSeq)

