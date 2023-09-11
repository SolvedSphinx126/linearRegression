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


def cost(t0, t1):
    cost = 0.0
    for i in range(0, m):
        cost += ((t1*x[i]+t0)-(y[i]))**2
    return (1/(2*m))*cost

def d0():
    sum = 0
    for i in range(0,m):
        sum += theta0 + theta1 * x[i] - y[i]
    return (1/m) * sum

def d1():
    sum = 0
    for i in range(0,m):
        sum += (theta0 + theta1 * x[i] - y[i]) * x[i]
    return (1/m) * sum

def change():
    return math.sqrt((-alpha * d0())**2+(-alpha*d1())**2)

while (change() >= epsilon):
    delta0 = d0()
    delta1 = d1()
    theta0 = theta0 - alpha * delta0
    theta1 = theta1 - alpha * delta1

print(f"The final error is {cost(theta0, theta1)}")

plt.scatter(x,y)
plt.plot(x, [theta0 + xi * theta1 for xi in x], color='red')
plt.show()