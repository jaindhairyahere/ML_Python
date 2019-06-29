import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import time
import pandas as pd
import copy
from scipy.special import expit as g				#Sigmoid Function

#########################################################################################
#Reading data into matrices X, y
def readFile(filename):
	datafile = open(filename,'r')
	lines	 = datafile.readlines()
	m		 = len(lines)
	n		 = len(lines[0].split(','))-1
	X		 = np.ones((m,n), order='F', dtype='float64')
	y		 = np.ones((m,1), order='F', dtype='float64')
	for i in range(0,m):
		s			= lines[i].split(',')
		for j in range(len(s)-1):
			X[i][j] 	= float(s[j])
		y[i]		= s[-1].rstrip()
	return [X,y,m,n]
[X,y,m,n]=readFile("ex2data2.txt")

#######################################################################################
#Preprocessing - Generating Polynomial Features
process = PolynomialFeatures(6)
X_processed=process.fit_transform(X)
n=X_processed.shape[1]
########################################################################################
#Calculating Cost Function
def costFunction(features,target,weights,reg_parameter):
	hypothesis =np.array(g(features.__matmul__(weights)))

	I=np.ones((m,1))
	cost=(-1)*(target.T.__matmul__(np.log(hypothesis)) + (I-target).T.__matmul__(np.log(I-hypothesis)))/m

	reg_func = (reg_parameter/(2*m))* (sum(weights**2) - weights[0][0]**2)

	return cost+reg_func

theta_ini=np.zeros((n,1))
print("Cost at theta =0 is : ",costFunction(X_processed,y,theta_ini,1))
#######################################################################################
#Testing
test_theta=np.ones((n,1))
print("Cost at test theta is : ",costFunction(X_processed,y,test_theta,10))




#######################################################################################
#Training Gradient DEscent ------- Passed
def trainGradientDescent(features,target,num_steps,learning_rate,reg_parameter,add_intercept=False):
	if add_intercept:
		intercept=np.ones((features.shape[0],1))
		features =np.hstack([intercept,features])

	weights = np.zeros((features.shape[1],1))

	for step in range(int(num_steps)):
		weights += (learning_rate)*calcGradient(features,target,weights,reg_parameter)
	return weights

def calcGradient(features,target,weights,reg_parameter,add_intercept=False):
	if add_intercept:
		intercept=np.ones((features.shape[0],1))
		features =np.hstack([intercept,features])

	score = np.matmul(features,weights)
	hypothesis = g(score)

	output_error_signal= hypothesis - target
	gradient = np.matmul(features.T,output_error_signal)
	weights_zero = copy.deepcopy(weights)
	weights_zero[0][0]=0
	reg_gradient = gradient/m + (reg_parameter/m)*weights_zero
	return reg_gradient

Opt_theta =trainGradientDescent(X_processed,y,300000,5e-5,10)
print("Minimized Cost Function by GD 	 is :",costFunction(X_processed,y,Opt_theta.reshape(n,1),10))



########################################################################################
#Using Library Methods
regression=LogisticRegression(penalty='l2',solver='lbfgs',C=0.1,fit_intercept=False)
regression.fit(X_processed,y.reshape(m))
print("Optimum Theta by Library method is :",pd.DataFrame(regression.coef_))
print("Minimized Cost Function by LM   is :",costFunction(X_processed,y,regression.coef_.reshape(n,1),10))
##################################################################################
#Plotting the Graphs
x1=np.array([X_processed[i][1] for i in range(m)])
plt.figure()
for i in range(m):
	if y[i]==1:
		plt.plot([X_processed[i][1]],[X_processed[i][2]],'ro')
	elif y[i]==0:
		plt.plot([X_processed[i][1]],[X_processed[i][2]],'b+')


delta = 50
xrange = np.linspace(-1.0, 1.5, delta)
yrange = np.linspace(-1.0, 1.5, delta)
X, Y = np.meshgrid(xrange,yrange)

z=np.zeros((delta,delta))
for i in range(len(xrange)):
	for j in range(len(yrange)):
		a=process.fit_transform(np.array([X[0][i],Y[j][0]]).reshape(1,2))
		z[i][j]=np.matmul(a,Opt_theta)


plt.contour(X, Y, z.T)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Classification Problem")

plt.show()

