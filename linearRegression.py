# Import necessary libraries
import math
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

# Split the dataset
trainingData = (data.iloc[0:math.ceil(data.shape[0] * (percentTraining / 100))])
validationData = (data.iloc[math.ceil(data.shape[0] * (percentTraining / 100)):])

# Normalize the datasets
nTrainingData = trainingData.apply(lambda col : [(x - np.mean(col)) / (np.std(col) if np.std(col) != 0 else 1) for x in col], raw=True)
nValidationData = validationData.apply(lambda col : [(x - np.mean(col)) / (np.std(col) if np.std(col) != 0 else 1) for x in col], raw=True)

# Define the cost function for linear regression
def J(theta, x, y):
    sum = 0
    for i in range(0, len(x)):
        sum += (hypothesis(x[i], theta) - y[i]) ** 2
    return (1 / (2 * len(x))) * sum

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
    #print(f"Iteration Count: {iterationCount}")
    print("Gradient descent completed:")
    print(f"   Iterations  : {iterationCount}")
    print(f"   Final Cost  : {J(theta, ix, y)[0]}")
    print(f"   Convergence : (delta thetas) < (epsilon) : {calcConvergence(prevTheta, theta)[0]} < {epsilon}")
    #print(f"Delta Thetas : {calcConvergence(prevTheta, theta)}")
    return theta

#Calculates the squared error in the open form.
def meanSquaredError(x, y, theta):
    # Create a new list so as to not modify the one passed in
    ix = np.concatenate(((np.ones((x.shape[0], 1), dtype=x.dtype)), x), axis=1)
    sum = 0
    for i in range(len(ix)):
        sum += (hypothesis(ix[i], theta) - y[i]) ** 2
    return (1 / len(ix)) * sum

# Return the theta vector that best fits the data
def bestFit(x, y):
    # Create a new list so as to not modify the one passed in
    ix = np.concatenate(((np.ones((x.shape[0], 1), dtype=x.dtype)), x), axis=1)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(ix.T, ix)), ix.T), y)


###################################################################################################
#                                       START OF TEST CASES                                       #
###################################################################################################


outFile = open("Output.txt", "w")
outLines = []

#NOX theta prediction with DIS and RAD training sets
x = np.array(nTrainingData[["DIS", "RAD"]])
y = np.array(nTrainingData[["NOX"]])
print("Performing gradient descent with x = DIS, RAD and y = NOX")
theta = gradientDescent(x,y)

outLines.append("--- Predicting NOX based on DIS and RAD with gradient descent ---\n")
outLines.append(f"   Theta Vector               : {[theta[i][0] for i in range(len(theta))]}\n") # so that stuff prints on one line
outLines.append(f"   Mean Squared Error (TRAINING DATA)   : {meanSquaredError(x, y, theta)}\n")
x = np.array(nValidationData[["DIS", "RAD"]])
y = np.array(nValidationData[["NOX"]])
outLines.append(f"   Mean Squared Error (VALIDATION DATA) : {meanSquaredError(x, y, theta)}\n")
outLines.append("\n")
outFile.writelines(outLines)

#NOX open form squared error prediction with validation set of 2 variables
print(f"Open form Mean Squared Error, NOX with 2 variables:     {meanSquaredError(x, y, theta)}")
print()



#NOX predicition with all variables
x = np.array(nTrainingData[["CRIM", "ZN", "INDUS", "CHAS", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]])
y = np.array(nTrainingData[["NOX"]])
print("Performing gradient descent with x = CRIM, ZN, INDUS, CHAS, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV and y = NOX")
theta = gradientDescent(x,y)

outLines = []
outLines.append("--- Predicting NOX based on CRIM, ZN, INDUS, CHAS, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, and MEDV with gradient descent ---\n")
outLines.append(f"   Theta Vector               : {[theta[i][0] for i in range(len(theta))]}\n") # so that stuff prints on one line
outLines.append(f"   Mean Squared Error (TRAINING DATA)   : {meanSquaredError(x, y, theta)}\n")
x = np.array(nValidationData[["CRIM", "ZN", "INDUS", "CHAS", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]])
y = np.array(nValidationData[["NOX"]])
outLines.append(f"   Mean Squared Error (VALIDATION DATA) : {meanSquaredError(x, y, theta)}\n")
outLines.append("\n")
outFile.writelines(outLines)

#NOX open form squared error prediction with validation set of all variables
print(f"Open form Mean Squared Error, NOX with all variables:     {meanSquaredError(x, y, theta)}")
print()



#MEDV predicition with AGE and TAX
x = np.array(nTrainingData[["AGE", "TAX"]])
y = np.array(nTrainingData[["MEDV"]])
print("Performing gradient descent with x = AGE, TAX and y = MEDV")
theta = gradientDescent(x,y)


outLines = []
outLines.append("--- Predicting MEDV based on AGE and TAX with gradient descent ---\n")
outLines.append(f"   Theta Vector               : {[theta[i][0] for i in range(len(theta))]}\n") # so that stuff prints on one line
outLines.append(f"   Mean Squared Error (TRAINING DATA)   : {meanSquaredError(x, y, theta)}\n")
x = np.array(nValidationData[["AGE", "TAX"]])
y = np.array(nValidationData[["MEDV"]])
outLines.append(f"   Mean Squared Error (VALIDATION DATA) : {meanSquaredError(x, y, theta)}\n")
outLines.append("\n")
outFile.writelines(outLines)

#MEDV open form squared error prediction with validation set of 2 variables
x = np.array(nValidationData[["AGE", "TAX"]])
y = np.array(nValidationData[["MEDV"]])
print(f"Open form Mean Squared Error, MEDV with 2 variables:     {meanSquaredError(x, y, theta)}")
print()



#MEDV prediction with all variables
x = np.array(nTrainingData[["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]])
y = np.array(nTrainingData[["MEDV"]])
print("Performing gradient descent with x = CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT y = MEDV")
theta = gradientDescent(x,y)


outLines = []
outLines.append("--- Predicting MEDV based on CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, and LSTAT with gradient descent ---\n")
outLines.append(f"   Theta Vector               : {[theta[i][0] for i in range(len(theta))]}\n") # so that stuff prints on one line
outLines.append(f"   Mean Squared Error (TRAINING DATA)   : {meanSquaredError(x, y, theta)}\n")
x = np.array(nValidationData[["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]])
y = np.array(nValidationData[["MEDV"]])
outLines.append(f"   Mean Squared Error (VALIDATION DATA) : {meanSquaredError(x, y, theta)}\n")
outLines.append("\n")
outFile.writelines(outLines)

#MEDV open form squared error prediction with validation set of all variables
x = np.array(nValidationData[["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]])
y = np.array(nValidationData[["MEDV"]])
print(f"Open form Mean Squared Error, MEDV with 2 variables:     {meanSquaredError(x, y, theta)}")
print()


# NOX prediction with DIS and RAD using closed form solution
x = np.array(nTrainingData[["DIS", "RAD"]])
y = np.array(nTrainingData[["NOX"]])
theta = bestFit(x, y)

outLines = []
outLines.append("--- Predicting NOX based on DIS and RAD with closed form solution ---\n")
outLines.append(f"   Theta Vector               : {[theta[i][0] for i in range(len(theta))]}\n") # so that stuff prints on one line
outLines.append(f"   Mean Squared Error (TRAINING DATA)   : {meanSquaredError(x, y, theta)}\n")
x = np.array(nValidationData[["DIS", "RAD"]])
y = np.array(nValidationData[["NOX"]])
outLines.append(f"   Mean Squared Error (VALIDATION DATA) : {meanSquaredError(x, y, theta)}\n")
outLines.append("\n")
outFile.writelines(outLines)


# MEDV prediction with AGE and TAX using closed form solution
x = np.array(nTrainingData[["AGE", "TAX"]])
y = np.array(nTrainingData[["MEDV"]])
theta = bestFit(x, y)

outLines = []
outLines.append("--- Predicting MEDV based on AGE and TAX with closed form solution ---\n")
outLines.append(f"   Theta Vector               : {[theta[i][0] for i in range(len(theta))]}\n") # so that stuff prints on one line
outLines.append(f"   Mean Squared Error (TRAINING DATA)   : {meanSquaredError(x, y, theta)}\n")
x = np.array(nValidationData[["AGE", "TAX"]])
y = np.array(nValidationData[["MEDV"]])
outLines.append(f"   Mean Squared Error (VALIDATION DATA) : {meanSquaredError(x, y, theta)}\n")
outLines.append("\n")
outFile.writelines(outLines)

print("\n==================")
print("WROTE DATA TO FILE")
print("==================\n")

outFile.close()