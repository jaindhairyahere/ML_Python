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
	X		 = datafile['X']
	y		 = datafile['y']
	m		 = len(datafile['X'])
	n		 = len(datafile['X'][0])
	Y		 = np.zeros((5000,10))
	for i in range(m):
		a	= y[i][0]
		if a==10:
			a=0
		Y[i][a]= 1
	return [X,Y,m,n]
[X,Y,m,n]	 = readFile("ex3data1.mat")

########################################################################################
#Pre-Processing

print('Shape of X : ',X.shape)
print('Shape of Y : ',Y.shape)

########################################################################################
#Calculating Cost Function
def costFunction(features,target,weights,reg_parameter):
	hypothesis =np.array(g(np.matmul(features,weights)))
	I=np.ones(target.shape)
	a = np.matmul(target.T,np.log(hypothesis))
	b = np.matmul((I-target).T,np.log(I-hypothesis))

	cost=(-1)*(sum(a)+sum(b))/m
	reg_func = (reg_parameter/(2*m))* (sum(weights**2) - weights[0][0]**2)

	a = max(cost)
	return a+max(reg_func)

theta_ini=np.zeros((n,10))
print("Cost at theta =0 is : ",costFunction(X,Y,theta_ini,1))
#######################################################################################
#Testing
# test_theta=np.ones((n,10))
# print("Cost at test theta is : ",costFunction(X,Y,test_theta,10))




#######################################################################################
#Training Gradient DEscent ------- Passed

def trainGradientDescent(features,target,num_steps,learning_rate,reg_parameter,add_intercept=False):
	if add_intercept:
		intercept=np.ones((features.shape[0],1))
		features =np.hstack([intercept,features])

	weights = np.zeros((features.shape[1],10))

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

Opt_theta =trainGradientDescent(X,Y,300000,5e-5,10)
print(Opt_theta)
print("Minimized Cost Function by GD 	 is :",costFunction(X,Y,Opt_theta.reshape(n,1),10))



########################################################################################
#Using Library Methods
regression=LogisticRegression(penalty='l2',solver='lbfgs',C=0.1,fit_intercept=False)
regression.fit(X_processed,y.reshape(m))
print("Optimum Theta by Library method is :",pd.DataFrame(regression.coef_))
print("Minimized Cost Function by LM   is :",costFunction(X,Y,regression.coef_.reshape(n,1),10))

##################################################################################
#Plotting the Graphs
x1=np.array([X_processed[i][1] for i in range(m)])
plt.figure()
for i in range(m):
	if y[i]==1:
		plt.plot([X[i][1]],[X[i][2]],'ro')
	elif y[i]==0:
		plt.plot([X[i][1]],[X[i][2]],'b+')


delta = 50
xrange = np.linspace(-1.0, 1.5, delta)
yrange = np.linspace(-1.0, 	1.5, delta)
X, Y = np.meshgrid(xrange,yrange)

z=np.zeros((delta,delta))
for i in range(len(xrange)):
	for j in range(len(yrange)):
		a=process.fit_transform(np.array([X[0][i],Y[j][0]]).reshape(1,2))
		z[i][j]=np.matmul(a,Opt_theta)

plt.contour(X, Y, (z.T), [6])

plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Classification Problem")

plt.show()
