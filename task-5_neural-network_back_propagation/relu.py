import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def t_derivative(x):
    return np.ones(x.shape) - np.tanh(x) ** 2


data = np.loadtxt('data/dane7.txt')

X = data[:, 0]
X = X.reshape((1, len(X)))

y = data[:, 1]
y = y.reshape((1, len(y)))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

S1 = 100

W1 = np.random.random_sample((S1, 1))

B1 = (5 + 0.5) * np.random.random_sample((S1, 1)) - 0.5

W2 = np.random.random_sample((1, S1))

B2 = (5 + 0.5) * np.random.random_sample((1, 1)) - 0.5

# W1 = np.random.random_sample(len(X))
# B1 = (5 + 0.5) * np.random.random_sample(len(X)) - 0.5
# W2 = np.random.random_sample(len(X))
# B2 = (5 + 0.5) * np.random.random_sample(len(X)) - 0.5


epochs = 2000
lr = 0.001
final_output = None
for _ in range(epochs):

    A1 = np.maximum(X, 0)  # ReLu
    A2 = (W2 @ A1.T) + B2

    # back propagation
    E2 = y - A2
    E1 = np.transpose(W2) @ E2

    dW2 = lr * E2 @ np.transpose(A1)
    dB2 = lr * E2 @ np.ones(np.transpose(E2).shape)
    dW1 = lr * (np.divide(np.exp(X), np.exp(X) + 1)) * E1 @ np.transpose(X)
    dB1 = lr * (np.divide(np.exp(X), np.exp(X) + 1)) * E1 @ np.ones(np.transpose(X).shape)

    W2 = W2 + dW2
    B2 = B2 + dB2

    W1 = W1 + dW1
    B1 = B1 + dB1
    final_output = A2

    if epochs % 3 == 0:
        plt.clf()
        plt.plot(X, y, 'r*')
        plt.plot(X, A2)
        plt.pause(0.05)

plt.plot(X,y, 'r*')
plt.plot(X, final_output)

# print(X, y)
# plt.plot(X[0], y[0])
# plt.show()
# plt.plot(X[0], final_output[0])
# plt.show()
#
# W1 = np.random.random_sample(len(X))
# B1 = (5 + 0.5) * np.random.random_sample(len(X)) - 0.5
# W2 = np.random.random_sample(len(X))
# B2 = (5 + 0.5) * np.random.random_sample(len(X)) - 0.5
#
# lr = 0.001
# epochs = 1
# for _ in range(epochs):
#     errors = []
#     for xi, yi in zip(X, y):
#         N1 = tanh(xi * W1 + B1)
#         N2 = N1 @ W2 + B2
#         E2 = yi - N2
#         E1 =
#         # errors.append(E)
#
#     errors = sum(errors)


# plt.plot(X, y, "ro")
# plt.show()
#
# plt.plot(X_train, y_train, "bo")
# plt.show()
#
# plt.plot(X_test, y_test, "bo")
# plt.show()

# print(W1, W1.shape)
# print(W2, W2.shape)
# print(data)
# print(data.T)
