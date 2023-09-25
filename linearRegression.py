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


# Read data from the "data.txt" file and populate data list
data = pd.read_csv("data.csv")
means = []
stddevs = []
for col in data:
    for row in data.index.values:
        data[col][row] = (data[col][row] - np.mean(data[col])) / np.std(data[col])

data.to_csv("normalized_data.csv")

#x = data[["NOX"]]
#y = data[["DIS"], ["RAD"]]

# Define the cost function for linear regression
def J(theta, x, y, m):
    return ((1/(2*m)) * np.sum(hypothesis(theta, x) - y) ** 2)

# Define the hypothesis function for linear regression
def hypothesis(theta, x):
    return np.matmul(theta.T, x)[0][0]

# TODO: Find Gradient of all vars
# Calculate the gradient for theta0 (intercept)
def d0():
    sum = 0
    for i in range(0,m):
        sum += hypothesis(x[i]) - y[i]
    return (1/m) * sum

# Calculate the gradient for theta1 (slope)
def d1():
    sum = 0
    for i in range(0,m):
        sum += (hypothesis(x[i]) - y[i]) * x[i]
    return (1/m) * sum

# Calculate the convergence parameter for gradient descent
def convergenceParam():
    return math.sqrt((-alpha * d0())**2 + (-alpha * d1())**2)

t = np.array([[1],[2],[3]])
x = np.array([[4],[5],[6]])
print(hypothesis(t, x))
