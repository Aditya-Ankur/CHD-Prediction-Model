import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import math
import time
from utils import *

# Load data
df = pd.read_csv('heart.csv')

# Drop rows with missing values
df = df.dropna()

isMale = df['male'].to_numpy()
age = df['age'].to_numpy()
education = df['education'].to_numpy()
currentSmoker = df['currentSmoker'].to_numpy()
cigsPerDay = df['cigsPerDay'].to_numpy()
BPMeds = df['BPMeds'].to_numpy()
prevalentStroke = df['prevalentStroke'].to_numpy()
prevalentHyp = df['prevalentHyp'].to_numpy()
diabetes = df['diabetes'].to_numpy()
totChol = df['totChol'].to_numpy()
sysBP = df['sysBP'].to_numpy()
diaBP = df['diaBP'].to_numpy()
bmi = df['BMI'].to_numpy()
heartRate = df['heartRate'].to_numpy()
glucose = df['glucose'].to_numpy()

outcome = df['TenYearCHD'].to_numpy()

X = np.column_stack((isMale, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, bmi, heartRate, glucose))
Y = outcome

# Split data into training and testing sets
def split_data(X:list, Y:list, train_ratio:float) -> tuple:
    m = X.shape[0]
    train_size = int(train_ratio * m)
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.8)

# Normalize data
def zscore(X:list) -> list:
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train_norm = zscore(X_train)
X_test_norm = zscore(X_test)

# polynomial features with degree 2 without xi.xj terms for all i != j
def poly_features(X:list, degree:int) -> list:
    X_poly = np.ones((X.shape[0], 1))
    for i in range(1, degree+1):
        X_poly = np.column_stack((X_poly, X**i))
    return X_poly

X_train_poly = poly_features(X_train_norm, 2)
X_test_poly = poly_features(X_test_norm, 2)

# Logistic regression
def f_wb(X:list, w:list, b:float) -> list:
    return 1/(1 + np.exp(-np.dot(X, w) - b))

def costFunction(X:list, Y:list, w:list, b:float, lmbda:float) -> list:
    m = X.shape[0]
    loss = -(np.sum(Y*np.log(f_wb(X, w, b)) + (1-Y)*np.log(1 - f_wb(X, w, b))))
    reg = (lmbda/2) * np.sum(w**2) # regularization term
    return (loss + reg) / m

def gradients(X:list, Y:list, w:list, b:float, lmbda:float) -> tuple:
    m,n = X.shape
    dw = np.zeros(n)
    db = 0
    for i in range(m):
        db += (f_wb(X[i], w, b) - Y[i])
        for j in range(n):
            dw[j] += (f_wb(X[i], w, b) - Y[i]) * X[i,j]
    # regularization term
    for j in range(n):
        dw[j] = (dw[j] + lmbda*w[j]) / m
    
    return dw, db/m

def gradientDescent(X:list, Y:list, w_ini:list, b_ini:float, alpha:float, num_iters:int, lmbda:float) -> tuple:
    w = copy.deepcopy(w_ini)
    b = copy.deepcopy(b_ini)
    costs = []
    for i in range(num_iters):
        dw, db = gradients(X, Y, w, b, lmbda)
        w -= alpha * dw
        b -= alpha * db

        if i<100000:      # prevent resource exhaustion 
            costs.append(costFunction(X, Y, w, b, lmbda))
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            print(f"Iteration {i}:                   Cost {float(costs[-1])}   ")
    return w, b, costs

# Initialize weights and bias
w_ini = np.zeros(X_train_poly.shape[1])
b_ini = 0
alpha = 0.01
num_iters = 1000
lmbda = 0.5

t0 = time.
w, b, costs = gradientDescent(X_train_poly, Y_train, w_ini, b_ini, alpha, num_iters, lmbda)

# Predict
p = f_wb(X_test_poly, w, b)
pred = np.array([1 if i > 0.5 else 0 for i in p])

# Evaluate
accuracy = np.mean(pred == Y_test)

print(f"w final: {w}")
print(f"b final: {b}")
print(f"Accuracy: {accuracy}")

# w final: [-0.35537872  0.21526774  0.36906146 -0.01625272  0.0704843   0.21089957
#   0.06134054  0.02394321  0.21799754  0.05576988  0.08616593  0.23335728
#   0.07776204  0.08934986  0.01282563  0.1015581  -0.30347083 -0.09397473
#  -0.12803139 -0.35306429 -0.05626257 -0.00925771  0.00130889 -0.16963971
#  -0.00800824  0.0204693  -0.00540305  0.0294754  -0.0087156  -0.05218412
#   0.0139809 ]
# b final: -0.3558323717701009