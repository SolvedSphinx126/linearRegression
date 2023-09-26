# Import necessary libraries
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initialize empty lists and variables
m = 0  # Number of data points
epsilon = 0.000001  # Convergence threshold for gradient descent
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
        sum += (hypothesis(x[i], theta) - y[i]) ** 2
    return (1 / (2 * m)) * sum


# Define the hypothesis function for linear regression
def hypothesis(x, theta):
    x = x.reshape((2, 1))
    return np.matmul(theta.T, x)[0][0]


# Calculate the gradient for theta (slope)
def gradient(theta, x, y):
    offsets = []
    locm = len(x)
    for i in range(0, len(theta)):
        offsets.append(-alpha * (1 / locm) * np.sum([(hypothesis(x[j], theta) - y[j]) * x[j][i] for j in range(1, locm)]))
    return offsets

# Return the calculated squared difference between the old and new theta
def calcConvergence(oldThetas, newThetas):
    sum = 0
    for i in range(0, len(oldThetas)):
        sum += (newThetas[i] - oldThetas[i]) ** 2
    return sum ** 0.5


prevTheta = np.zeros((2, 1))
x = np.array(data[["DIS", "RAD"]])
y = np.array(data[["NOX"]])

# Calculate the first delta theta before entering the loop
theta = np.zeros((2, 1))
offsets = gradient(theta, x, y)
for i in range(0,len(theta)):
    theta[i] += offsets[i]

# Perform gadient decent until accuracy reached
while calcConvergence(prevTheta, theta) > epsilon:
    # Store the current theta before updating it
    for i in range(0, len(theta)):
        prevTheta[i] = theta[i]
    # Update the thetas
    offsets = gradient(theta, x, y)
    for i in range(0,len(theta)):
        theta[i] += offsets[i]
    print(f"Cost         : {J(theta, x, y, len(x))}")
    print(f"Delta Thetas : {calcConvergence(prevTheta, theta)}")


