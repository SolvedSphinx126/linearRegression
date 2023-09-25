# Import necessary libraries
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initialize empty lists and variables
m = 0  # Number of data points
epsilon = 0.01  # Convergence threshold for gradient descent
alpha = 0.01  # Learning rate for gradient descent
theta = np.array([])  # column vector of hypothesis

# Read data from the "data.txt" file and populate data list with normalized data
data = pd.read_csv("data.csv")
means = []
stddevs = []
for col in data:
    for row in data.index:
        data[col][row] = (data[col][row] - np.mean(data[col])) / np.std(data[col])

data.to_csv("normalized_data.csv")


# x = data[["NOX"]]
# y = data[["DIS"], ["RAD"]]

# Define the cost function for linear regression
def J(theta, x, y, m):
    sum = 0
    for i in range(0, len(x)):
        sum += (hypothesis(x[i]) - y[i]) ** 2
    return (1 / (2 * m)) * sum


# Define the hypothesis function for linear regression
def hypothesis(x):
    x = x.reshape((2, 1))
    return np.matmul(theta.T, x)[0][0]


# Calculate the gradient for theta (slope)
def gradient(theta, x, y):
    offsets = []
    locm = len(x)
    for i in range(0, len(theta)):
        offsets.append(-alpha * (1 / locm) * np.sum([(hypothesis(x[j]) - y[j]) * x[j][i] for j in range(1, locm)]))
    return offsets


theta = np.zeros((2, 1))
x = np.array(data[["DIS", "RAD"]])
y = np.array(data[["NOX"]])
for _ in range(1, 100):
    offsets = gradient(theta, x, y)
    for i in range(0,len(theta)):
        theta[i] += offsets[i]
    print(J(theta, x, y, len(x)))
