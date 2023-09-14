import math
import matplotlib.pyplot as plt

x = []
y = []
m = 0
epsilon = 0.01
alpha = 0.01
theta0 = 0
theta1 = 0

for line in open("data.txt"):
    [a, b] = [float(x) for x in line.split(',')]
    x.append(a)
    y.append(b)
    m+=1

def leastSquares():
    sumX = sum(x)
    sumY = sum(y)
    sumXY = sum([xi * yi for xi , yi in zip(x, y)])
    sumX2 = sum([xi ** 2 for xi in x])
    slope = (m * sumXY - sumX * sumY)/(m * sumX2 - sumX ** 2)
    offset = (sumY - slope * sumX)/m
    print(f"slope = {slope}, offset = {offset}")

def J(t0, t1):
    cost = 0.0
    for i in range(0, m):
        cost += (h(x[i])-(y[i]))**2
    return (1/(2*m))*cost

def h(x):
    return theta0 + theta1 * x

def d0():
    sum = 0
    for i in range(0,m):
        sum += h(x[i]) - y[i]
    return (1/m) * sum

def d1():
    sum = 0
    for i in range(0,m):
        sum += (h(x[i]) - y[i]) * x[i]
    return (1/m) * sum

def change():
    return math.sqrt((-alpha * d0())**2 + (-alpha * d1())**2)

while (change() >= epsilon):
    delta0 = d0()
    delta1 = d1()
    theta0 = theta0 - alpha * delta0
    theta1 = theta1 - alpha * delta1

    print(f"Theta 0 is {theta0}")
    print(f"Theta 1 is {theta1}")

delta0 = d0()
delta1 = d1()
theta0 = theta0 - alpha * delta0
theta1 = theta1 - alpha * delta1

print(f"Theta 0 is {theta0}")
print(f"Theta 1 is {theta1}")

print(f"The final error is {J(theta0, theta1)}")
leastSquares()

plt.scatter(x, y, color='red', marker='x', alpha=0.5)
plt.plot(x, [theta0 + xi * theta1 for xi in x], color='blue', linewidth=0.5, alpha=0.5)
plt.show()