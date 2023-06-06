import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def error(prediction, actual):
    actual = actual.T[0]
    prediction = prediction.T[0]
    return sum((actual - prediction) ** 2) / len(actual)

class AutoRegModel:
    def __init__(self):
        self.weights = None

    def fit(self, x_train, y_train):
        # Perform least squares regression to estimate the weights
        c = np.hstack([np.ones((len(x_train), 1)), x_train])
        self.weights = np.linalg.pinv(c) @ y_train

    def predict(self, x_test):
        # Make predictions using the estimated weights
        c = np.hstack([np.ones((len(x_test), 1)), x_test])
        return c @ self.weights

class RNNModel:
    def __init__(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(3, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=25, batch_size=1, verbose=2)

    def predict(self, x_test):
        return self.model.predict(x_test)

a = np.loadtxt('danet.txt')
x = a[0:40, [1, 2, 3]]
y = a[0:40, [0]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

ar = AutoRegModel()
ar.fit(x_train, y_train)
ar_prediction = ar.predict(x_test)

rnn = RNNModel()
rnn.fit(x_train.reshape(-1, 3, 1), y_train)
rnn_prediction = rnn.predict(x_test.reshape(-1, 3, 1))

plt.plot(ar_prediction, 'b-', label="Linear autoregression")
plt.plot(rnn_prediction, 'k-', label="RNN")
plt.plot(y_test, 'r-', label="Actual")
plt.legend()
plt.show()

ar_error = error(prediction=ar_prediction, actual=y_test)
rnn_error = error(prediction=rnn_prediction, actual=y_test)
mse_ar = mean_squared_error(y_true=y_test, y_pred=ar_prediction)
mse_rnn = mean_squared_error(y_true=y_test, y_pred=rnn_prediction)

print(f'AR model error: {ar_error}')
print(f'RNN model error: {rnn_error}')
print(f'AR model MSE: {mse_ar}')
print(f'RNN model MSE: {mse_rnn}')
