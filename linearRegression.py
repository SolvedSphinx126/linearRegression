# Import necessary libraries
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initialize empty lists and variables
m = 0   # Number of data points
epsilon = 0.01  # Convergence threshold for gradient descent
alpha = 0.01    # Learning rate for gradient descent
theta = np.array([])    # column vector of hypothesis


# Read data from the "data.txt" file and populate data list with normalized data
data = pd.read_csv("data.csv")
means = []
stddevs = []
for col in data:
    for row in data.index.values:
        data.loc[:, (col, row)] = (data.loc[:, (col, row)] - np.mean(data[col])) / np.std(data[col])

data.to_csv("normalized_data.csv")

#x = data[["NOX"]]
#y = data[["DIS"], ["RAD"]]

# Define the cost function for linear regression
def J(theta, x, y, m):
    return ((1/(2*m)) * np.sum(hypothesis(theta, x) - y) ** 2)

# Define the hypothesis function for linear regression
def hypothesis(x):
    return np.matmul(theta.T, x)[0][0]

# Calculate the gradient for theta (slope)
def gradient(theta, x, y):
    offsets = []
    m=len(x)
    for i in range(0,len(theta)):
        offsets.append(-alpha*(1/m)*np.sum([hypothesis(x.values[j])-y[j]*x[j][i] for j in range(0,m)]))
        return offsets

theta = np.array([0],[0])
x = data[["DIS","RAD"]]
y = data[["NOX"]]
for _ in range(1,20):
    offsets = gradient(theta, x, y)
    for i in len(theta):
        theta[i] += offsets[i]
    print(J(theta, x, y, len(x)))

