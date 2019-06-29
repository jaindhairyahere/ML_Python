import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from scipy.special import expit as g				#Sigmoid Function

def readData(filename):
    X = pd.read_csv(filename,index_col=0)
    return X
X = readData("train.csv")
