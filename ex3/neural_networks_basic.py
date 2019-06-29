import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import copy
from scipy.special import expit as g				#Sigmoid Function
from scipy.io import loadmat
#########################################################################################
#Reading data into matrices X, y ,Y
def readFile(filename):
	datafile = loadmat(filename)
	weights  = loadmat('ex3weights.mat')
	Theta1	 = weights["Theta1"]
	Theta2	 = weights["Theta2"]
	X		 = datafile['X']
	y		 = datafile['y']
	m		 = len(datafile['X'])
	n		 = len(datafile['X'][0])
	Y		 = np.zeros((5000,10))
	X 		 = np.hstack([np.zeros([5000,1]),X])
	for i in range(m):
		a	= y[i][0]
		if a==10:
			a=0
		Y[i][a]= 1
	return [X,Y,m,n,Theta1,Theta2]
[X,Y,m,n,Theta1,Theta2]	 = readFile("ex3data1.mat")
print(Y.shape)

print("Initial Layer size (Layer 1) : ",len(X[0]))
print("Initial Layer size (Layer 1) : ",Theta1.size/len(Theta1) -1)
print("Hidden  Layer size (Layer 2) : ",Theta2.size/len(Theta2) -1)
print("Final   Layer size (Layer 3) : ",len(Theta2))
print("Theta 1 shape 	  (Layer 1) : ",Theta1.shape)
print("Theta 2 shape 	  (Layer 2) : ",Theta2.shape)


def solveNeuralNetworks():
	print(" X shape = ",X.shape,"\n Theta1 shape =",Theta1.shape)
	z2 = np.matmul(X,Theta1.T)
	a2 = g(z2)
	print(" a2 shape = ",a2.shape,"\n Theta2 shape =",Theta2.shape)
	a2 = np.hstack([np.zeros([5000,1]),a2])
	z3 = np.matmul(a2,Theta2.T)
	h  = g(z3)
	predict = np.argmax(h.T,axis=0)
#	print(np.argmax(Y,axis=1))
solveNeuralNetworks()
