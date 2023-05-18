import numpy as np
import matplotlib.pyplot as plt
import time

P = np.arange(-4, 4.1, 0.1)
T = P**2 + 1 * (np.random.rand(len(P)) - 0.5)

S1 = 2
W1 = np.random.rand(S1, 1) - 0.5
B1 = np.random.rand(S1, 1) - 0.5
W2 = np.random.rand(1, S1) - 0.5
B2 = np.random.rand(1, 1) - 0.5
lr = 0.001

for epoka in range(1, 201):
    X = np.dot(W1, P.reshape(1, -1)) + np.dot(B1, np.ones_like(P).reshape(1, -1))
    A1 = np.maximum(X, 0)
    A2 = np.dot(W2, A1) + B2

    E2 = T - A2
    E1 = np.dot(W2.T, E2)

    dW2 = lr * np.dot(E2, A1.T)
    dB2 = lr * np.dot(E2, np.ones_like(E2).T)
    dW1 = lr * (np.exp(X) / (np.exp(X) + 1)) * np.dot(E1, P.reshape(1, -1))
    dB1 = lr * (np.exp(X) / (np.exp(X) + 1)) * np.dot(E1, np.ones_like(P).reshape(1, -1))

    W2 = W2 + dW2
    B2 = B2 + dB2
    W1 = W1 + dW1
    B1 = B1 + dB1

    if epoka % 5 == 0:
        plt.clf()
        plt.plot(P, T, 'r*')
        plt.plot(P, A2)
        plt.pause(0.05)

plt.show()
