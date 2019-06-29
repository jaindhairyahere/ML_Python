import matplotlib.pyplot as plotter
import scipy.linalg as alg
import numpy
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_squared_error, r2_score
import time
import copy
from scipy.special import expit as sigmoid				#Sigmoid Function

#########################################################################################
#Reading data into matrices X	, y
datafile = open("ex2data1.txt",'r')
lines	 = datafile.readlines()
m		 = len(lines)
fm		 = [[1]*len(lines[0].split(',')) for k in range(m)]
y		 = [0]*m
for i in range(0,m):
	s			= lines[i].split(',')
	for j in range(len(s)-1):
		fm[i][j+1] 	= float(s[j])
	y[i]		= s[-1].rstrip()
X=numpy.array(fm,order='F')
X=X.astype(numpy.float64)
y=numpy.array(y,order='F')
y=y.astype(numpy.float64)

########################################################################################
#Using Gradient Descent and Normal Equation
theta_ini	=numpy.array([0 for k in range(len(X[0]))],order='F')
num_iter	=1500
alpha		=0.01
theta_iter	=[]
h1=[]
J1=[]

#######################################################################################
#Calculating cost
def costFunction(X,theta,y):
	h=sigmoid(X.__matmul__(theta))
	I=numpy.array([1 for i in range(len(lines))])
	J=(-1)*(y.__matmul__(numpy.log(h)) + (I-y).__matmul__(numpy.log(I-h)))/len(lines)
	return J

def trainGradientDescent(alpha,num_iter,theta_ini):
	for i in range(num_iter):
		p=numpy.array(sigmoid(X.__matmul__(theta_ini)) - y,order='F')
		h1.append(sigmoid(X.__matmul__(theta_ini)))
		theta_ini=theta_ini-(alpha/m)*(X.transpose().__matmul__(p))
		theta_iter.append(theta_ini)
		J1.append(costFunction(X,theta_ini,y))
	return theta_ini
theta_grad	= numpy.array(trainGradientDescent(alpha,num_iter,theta_ini))
h_GD=numpy.array(X.__matmul__(theta_grad))
print("Theta using Gradient descent : ",theta_grad)

print("Minimized Cost is : ",costFunction(X,numpy.array([-25.161272,0.206233,0.201470]),y))

#######################################################################################
#Predicting Outcomes
p=numpy.array([0 for k in range(len(y))],order='F')
def calcPredict(X,theta):
	h=sigmoid(X.__matmul__(theta))
	for i in range(len(h)):
		if h[i]>0.5:
			p[i]=1
		else:
			p[i]=0
	return h

##############################################################################################
#Plotting Data on graph
plotter.figure()
fig, ax =plotter.subplots(2,1)
for i in range(len(lines)):
	if y[i]==1:
		ax[0].plot([X[i][1]],[X[i][2]],'ro')
	if y[i]==0:
		ax[0].plot([X[i][1]],[X[i][2]],'b+')
ax[0].plot([25.161272/0.206233,0],[0,25.161272/0.201470],'--')
ax[1].plot(J1)

plotter.show()
