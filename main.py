import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y = np.array([250, 300, 480, 430, 630, 730])
m = x.shape[0]

w = 0
b = 0

alpha = 0.01
iterations = 1000
cost_history = []

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    return cost / (2 * m)

def gradient_descent(x, y, w, b, alpha, iterations):
    m = x.shape[0]
    for _ in range(iterations):
        dw = 0
        db = 0
        for i in range(m):
            f_wb = w * x[i] + b
            dw += (f_wb - y[i]) * x[i]
            db += (f_wb - y[i])
        dw /= m
        db /= m
        w = w - alpha * dw
        b = b - alpha * db
        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)
    return w, b

w, b = gradient_descent(x, y, w, b, alpha, iterations)
print(f"Final weights: w = {w:.2f}, b = {b:.2f}")

plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Over Time")
plt.grid()
plt.show()

test_x = 2.0
predicted_price = w * test_x + b
print(f"Predicted price for 2000 sqft house: ${predicted_price:.2f}k")
