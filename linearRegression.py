# Import necessary libraries
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initialize empty lists and variables
m = 0  # Number of data points
epsilon = 0.001  # Convergence threshold for gradient descent
alpha = 0.01  # Learning rate for gradient descent
theta = np.array([])  # column vector of hypothesis
percentTraining = 90 # Percent of data to be used for training

# Open the file as read only
inFile = open("boston.txt", "r")
fileLines = inFile.readlines()
varLabels = []
rawDataVals = []

line = fileLines[0]
i = 0
# Skip over everything till the first blank line
while line != "\n":
    i += 1
    line = fileLines[i]
# Skip the blank line and the line announcing the variables
i += 2
line = fileLines[i]
# Loop over all the variable labels
while line != "\n":
    i += 1
    varLabels.append(line.split()[0])
    line = fileLines[i]
# Skip the new line
i += 1
# Loop over the data values and put them all into an array
for index in range(i, len(fileLines)):
    line = fileLines[index]
    for val in line.split():
        rawDataVals.append(float(val))

formattedDataVals = []
i = 0
# Turn the 1d array into a 2d array
while i < len(rawDataVals):
    formattedDataVals.append([rawDataVals[j] for j in range(i, i + len(varLabels))])
    i += len(varLabels)

data = pd.DataFrame(formattedDataVals, columns=varLabels, dtype=float)

# Normalize the dataset
nData = data.apply(lambda col : [(x - np.mean(col)) / np.std(col) for x in col], raw=True)

trainingData = nData.iloc[0:math.ceil(nData.shape[0] * (percentTraining / 100))]
validationData = nData.iloc[math.ceil(nData.shape[0] * (percentTraining / 100)):]

# Define the cost function for linear regression
def J(theta, x, y, m):
    sum = 0
    for i in range(0, len(x)):
        sum += (hypothesis(x[i], theta) - y[i]) ** 2
    return (1 / (2 * m)) * sum

# Define the hypothesis function for linear regression
def hypothesis(x, theta):
    x = x.reshape((len(x), 1))
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

# Return theta vector trained by gradient descent
def gradientDescent(x, y):
    theta = np.zeros((len(x[0]) + 1, len(y[0])))
    # Create a new list so as to not modify the one passed in
    ix = np.concatenate(((np.ones((x.shape[0], 1), dtype=x.dtype)), x), axis=1)

    prevTheta = theta.copy()
    offsets = gradient(theta, ix, y)
    for i in range(0, len(theta)):
        theta[i] += offsets[i]

    iterationCount = 0
    while calcConvergence(prevTheta, theta) > epsilon:
        iterationCount += 1
        # Store the current theta before updating it
        for i in range(0, len(theta)):
            prevTheta[i] = theta[i]
        # Update the thetas
        offsets = gradient(theta, ix, y)
        for i in range(0,len(theta)):
            theta[i] += offsets[i]
    print(f"Iteration Count: {iterationCount}")
    print(f"Cost         : {J(theta, ix, y, len(ix))}")
    print(f"Delta Thetas : {calcConvergence(prevTheta, theta)}")
    print()
    return theta

# Return the theta vector that best fits the data
def bestFit(x, y):
    # Create a new list so as to not modify the one passed in
    ix = np.concatenate(((np.ones((x.shape[0], 1), dtype=x.dtype)), x), axis=1)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(ix.T, ix)), ix.T), y)

#NOX prediction with DIS and RAD
x = np.array(trainingData[["DIS", "RAD"]])
y = np.array(trainingData[["NOX"]])
theta = gradientDescent(x,y)

#NOX predicition with all variables
x = np.array(trainingData[["CRIM", "ZN", "INDUS", "CHAS", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]])
theta = gradientDescent(x,y)

#MEDV predicition with AGE and TAX
x = np.array(trainingData[["AGE", "TAX"]])
y = np.array(trainingData[["MEDV"]])
theta = gradientDescent(x,y)

#MEDV prediction with all variables
x = np.array(trainingData[["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]])
theta = gradientDescent(x,y)
