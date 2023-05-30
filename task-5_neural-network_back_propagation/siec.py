import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def tanh_deriv(x):
    return np.ones(x.shape) - np.tanh(x) ** 2


data = np.loadtxt('data/dane7.txt')

X = data[:, 0]
X = X.reshape((1, len(X)))

y = data[:, 1]
y = y.reshape((1, len(y)))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

S1 = 100
W1 = np.random.random_sample((S1, 1))
# print("W1", W1)
B1 = (5 + 0.5) * np.random.random_sample((S1, 1)) - 0.5
# print("B1", B1)
W2 = np.random.random_sample((1, S1))
# print("W2", W2)
B2 = (5 + 0.5) * np.random.random_sample((1, 1)) - 0.5
# print("B2", B2)


epochs = 20000
lr = 0.0001
final_output = None
for _ in range(epochs):
    # print(f"epoch with index {_}")
    A1 = np.tanh(W1 @ X + B1)
    A2 = W2 @ A1 + B2

    # back propagation
    E2 = y - A2
    E1 = np.transpose(W2) @ E2

    dW2 = lr * E2 @ np.transpose(A1)
    dB2 = lr * E2 @ np.ones(np.transpose(E2).shape)
    dW1 = lr * tanh_deriv(A1) * E1 @ np.transpose(X)
    dB1 = lr * tanh_deriv(A1) * E1 @ np.ones(np.transpose(X).shape)

    W2 = W2 + dW2
    B2 = B2 + dB2

    W1 = W1 + dW1
    B1 = B1 + dB1

    final_output = A2

print(X, y)
plt.plot(X[0], y[0])
# plt.show()
plt.plot(X[0], final_output[0])
plt.show()


