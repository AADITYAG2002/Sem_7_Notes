import numpy as np

X = [0.5, 2.5]
Y = [0.2, 0.9]


def f(w, b, x):
    return 1.0 / (1.0 + np.exp(-(w * x) + b))


def error(w, b):
    err = 0.0
    for x, y in zip(X, Y):
        fx = f(w, b, x)
        err += 0.5 * (fx - y) ** 2
    return err


def grad_b(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx)


def grad_w(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx) * x


w, b, eta, max_epochs = -2, -2, 1.0, 1000
for _ in range(max_epochs):
    dw, db = 0, 0
    for x, y in zip(X, Y):
        dw += grad_w(w, b, x, y)
        db += grad_b(w, b, x, y)
    w = w - eta * dw
    b = b - eta * db

print(f"w : {w}, b : {b}")
