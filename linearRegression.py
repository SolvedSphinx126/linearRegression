# Import necessary libraries
import math
import matplotlib.pyplot as plt

# Initialize empty lists and variables
x = []  # List to store x-values from the data
y = []  # List to store y-values from the data
m = 0   # Number of data points
epsilon = 0.01  # Convergence threshold for gradient descent
alpha = 0.01    # Learning rate for gradient descent
theta0 = 0      # Initial guess for intercept
theta1 = 0      # Initial guess for slope

# Read data from the "data.txt" file and populate x and y lists
for line in open("data.txt"):
    [a, b] = [float(x) for x in line.split(',')]
    x.append(a)
    y.append(b)
    m+=1

# Define a function for the least squares method
def leastSquares():
    sumX = sum(x)
    sumY = sum(y)
    sumXY = sum([xi * yi for xi , yi in zip(x, y)])
    sumX2 = sum([xi ** 2 for xi in x])
    slope = (m * sumXY - sumX * sumY)/(m * sumX2 - sumX ** 2)
    offset = (sumY - slope * sumX)/m
    print()
    print("Least Squares Method results")
    print(f"slope / theta 1 = {slope}")
    print(f"y offset / theta 0 = {offset}")

# Define the cost function for linear regression
def J(t0, t1):
    cost = 0.0
    for i in range(0, m):
        cost += (h(x[i])-(y[i]))**2
    return (1/(2*m))*cost

# Define the hypothesis function for linear regression
def h(x):
    return theta0 + theta1 * x

# Calculate the gradient for theta0 (intercept)
def d0():
    sum = 0
    for i in range(0,m):
        sum += h(x[i]) - y[i]
    return (1/m) * sum

# Calculate the gradient for theta1 (slope)
def d1():
    sum = 0
    for i in range(0,m):
        sum += (h(x[i]) - y[i]) * x[i]
    return (1/m) * sum

# Calculate the convergence parameter for gradient descent
def convergenceParam():
    return math.sqrt((-alpha * d0())**2 + (-alpha * d1())**2)

# Perform gradient descent until convergence
prevChange = epsilon + 1
while (prevChange >= epsilon):
    prevChange = convergenceParam()
    delta0 = d0()
    delta1 = d1()
    theta0 = theta0 - alpha * delta0
    theta1 = theta1 - alpha * delta1

# Print the final values of theta0 and theta1
print(f"Theta 0 is {theta0}")
print(f"Theta 1 is {theta1}")

# Print the final error (cost) of the linear regression model
print(f"The final error is {J(theta0, theta1)}")

# Calculate estimates for x=3.5 and x=7 (35000 and 70000 population)
print()
x_estimates = [3.5, 7]
for x_val in x_estimates:
    y_estimate = h(x_val)
    print(f"Estimated profits for population={x_val * 10000:.0f} is ${y_estimate * 10000:.2f}")

# Perform the least squares calculation
leastSquares()

# Plot the data points in red and the regression line in blue
plt.scatter(x, y, color='red', marker='x', alpha=0.5)
plt.plot(x, [theta0 + xi * theta1 for xi in x], color='blue', linewidth=0.5, alpha=0.5)
plt.show()