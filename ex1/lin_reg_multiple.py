import matplotlib.pyplot as plotter
import scipy.linalg as alg
import numpy
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_squared_error, r2_score
import time
import copy
#########################################################################################
#Reading data into matrices X	, y
datafile = open("ex1data2.txt",'r')
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
################################################################################
#Perfrom Linear Regression using Libraries
reg			= linear_model.LinearRegression()
reg.fit(X,y)
theta_libr	= copy.deepcopy(reg.coef_)
theta_libr[0]=reg.intercept_
print("Theta Using Library Methods : ",theta_libr)
h_LM		= numpy.array(reg.predict(X),order='F')

#######################################################################################
#Computing Using Gradient Descent
theta_ini	=numpy.array([0 for k in range(len(X[0]))],order='F')
num_iter	=1500
alpha		=0.01
theta_iter	=[]
h1=[]
def trainGradientDescent(alpha,num_iter,theta_ini):
	for i in range(num_iter):
		p=numpy.array(X.__matmul__(theta_ini) - y,order='F')
		h1.append(X.__matmul__(theta_ini))
		theta_ini=theta_ini-(alpha/m)*(X.transpose().__matmul__(p))
		theta_iter.append(theta_ini)
	return theta_ini
theta_grad	= numpy.array(trainGradientDescent(alpha,num_iter,theta_ini))
h_GD=numpy.array(X.__matmul__(theta_grad))
print("Theta using Gradient descent : ",theta_grad)

a=alg.inv(numpy.array(X.transpose().__matmul__(X)))
b=X.transpose().__matmul__(y)
theta_ne=a.__matmul__(b)
h_NE=h_GD=numpy.array(X.__matmul__(theta_ne))
print("Theta Using Normal Equation : ",theta_ne)
#######################################################################################
#Computing Cost Function

def costFunction(X,y,theta,h):
	J=sum((h-y)*(h-y)/(2*len(lines)))
	return J
print()
print()
print("Cost Function Using Library  Methods: ",costFunction(X,y,theta_libr,h_LM))
print("Cost Function Using Gradient Descent: ",costFunction(X,y,theta_grad,h_GD))
print("Cost Function Using Normal  Equation: ",costFunction(X,y,theta_grad,h_NE))
print()
print()
#######################################################################################
#Plot Linear Regression fit and Graphs

print("Mean Squared Error : %.02f" %mean_squared_error(y,h_LM))
print("R^2 Score : ",r2_score(y,h_GD))
fig1=plotter.gca()
fig1.plot()
fig1.plot([costFunction(X,y,theta_iter[i],h1[i]) for i in range(num_iter)], [i for i in range(num_iter)])
plotter.ion()
plotter.ylabel('Cost Function')
plotter.xlabel('No. of iterations')
plotter.title('Costfunction vs num_iter plot')
plotter.axis('tight')
plotter.show()
time.sleep(5)
print("Clossing The Program. Exit Code 0")
plotter.close()
