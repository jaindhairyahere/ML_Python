import matplotlib.pyplot as plotter
import scipy.linalg as alg
import numpy
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_squared_error, r2_score
import time
import copy

###############################################################################
# Reading data into matrices X	, y

datafile = open("ex1data1.txt", 'r')
lines = datafile.readlines()
fm = [[1] * 2 for k in range(len(lines))]
y = [0] * len(lines)
for i in range(0, len(lines)):
    s = lines[i].split(',')
    fm[i][1] = float(s[0])
    y[i] = s[1].rstrip()
X = numpy.array(fm, order='F')
X = X.astype(numpy.float64)
y = numpy.array(y, order='F')
y = y.astype(numpy.float64)

##############################################################################
# Perfrom Linear Regression using Libraries
reg = linear_model.Ridge(alpha=0.04)
reg.fit(X, y)
theta_libr = copy.deepcopy(reg.coef_)
theta_libr[0] = reg.intercept_
print("Regularized Theta Using Library Methods : ", theta_libr)
h_LM = numpy.array(reg.predict(X), order='F')


##############################################################################
# Computing Using Gradient Descent
theta_ini = numpy.array([0, 0], order='F')
num_iter = 1500
alpha = 0.01
theta1_iter = []
theta0_iter = []
J1 = []


def trainGradientDescent(alpha, num_iter, theta_ini, al=0.04):
    for i in range(num_iter):
        p = numpy.array(X.__matmul__(theta_ini) - y, order='F')
        theta1_iter.append(theta_ini[1])
        theta0_iter.append(theta_ini[0])
        J1.append(X.__matmul__(theta_ini))
        theta_ini = theta_ini - (alpha / len(lines)) * \
            ((X.transpose().__matmul__(p)) + al * theta_ini)
    return theta_ini


theta_grad = numpy.array(trainGradientDescent(alpha, num_iter, theta_ini))
h_GD = numpy.array(X.__matmul__(theta_grad))
print("REgularized Theta using Gradient descent : ", theta_grad)

L = numpy.array([[0, 0], [0, 1]], order='F')
a = alg.inv(numpy.array(X.transpose().__matmul__(X) + 0.04 * L))
b = X.transpose().__matmul__(y)
theta_ne = a.__matmul__(b)
h_NE = numpy.array(X.__matmul__(theta_ne))
print("Regularized Theta Using Normal Equation : ", theta_ne)
##############################################################################
# Computing Cost Function


def costFunction(X, y, theta, h):
    J = (sum((h - y) * (h - y)) + theta.transpose().__matmul__(theta) -
         theta[0] * theta[0]) / (2 * len(lines))
    return J


print()
print(
)print("Regularized Cost Function Using Library  Methods: ",
      costFunction(X, y, theta_libr, h_LM))
print("Regularized Cost Function Using Gradient Descent: ",
      costFunction(X, y, theta_grad, h_GD))
print("Regularized Cost Function Using Normal  Equation: ",
      costFunction(X, y, theta_grad, h_NE))
print()
print()
print("Regularized Cost at theta=[0,0] hence h=[0...] : ", costFunction(
    X, y, numpy.array([0, 0], order='F'), [0 * j for j in range(len(lines))]))
print("Regularized Cost at theta=[-1,2] hence h=[2...] : ", costFunction(
    X, y, numpy.array([-1, 2], order='F'), X.__matmul__([-1, 2])))
print()
###############################################################################
# Plot Linear Regression fit and Graphs

print("Mean Squared Error : %.02f" % mean_squared_error(y, h_LM))
print("R^2 Score : ", r2_score(y, h_GD))
fig1 = plotter.gca()
fig1.plot()
fig1.plot(numpy.array([X[i][1] for i in range(len(lines))]), y, 'rx')
fig1.plot(numpy.array([X[i][1] for i in range(len(lines))]), h_GD)
plotter.ion()
plotter.xlabel('data')
plotter.ylabel('values')
plotter.title('Regularized Linear Regression Plot')
plotter.axis('tight')
plotter.show()
time.sleep(5)
plotter.close()
