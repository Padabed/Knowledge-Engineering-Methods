import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class MultiClassLogisticRegression(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def predict_proba_class(self, X, class_index):
        scores = self.net_input(X, self.W_[class_index])
        proba = self.activation(scores)
        # Print just the first 3 probabilities
        print("Probabilities for Class {}: {}".format(class_index, proba[:3]))
        return proba

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.W_ = []

        for k in np.unique(y):
            # Convert target variable to binary representation for the current class
            y_binary = np.where(y == k, 1, 0)
            # Initialize weights
            w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

            for _ in range(self.n_iter):
                # Compute the net input
                net_input = self.net_input(X, w)
                # Apply the activation function
                output = self.activation(net_input)
                # Compute the errors
                errors = y_binary - output
                # Update the weights using gradient descent
                w[1:] += self.eta * X.T.dot(errors)
                w[0] += self.eta * errors.sum()

            self.W_.append(w)

        return self

    def net_input(self, X, w):
        return np.dot(X, w[1:]) + w[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        scores = np.array([self.net_input(X, w) for w in self.W_])
        return np.argmax(scores, axis=0)


def acc(realState, predictedState):
    return np.sum(realState == predictedState) / len(predictedState)


def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Initialize and train the logistic regression classifier
    clf = MultiClassLogisticRegression(eta=0.28, n_iter=8000)
    clf.fit(X_train, y_train)

    # Evaluate the accuracy of the classifier
    predictions = clf.predict(X_test)
    print(acc(y_test, predictions))

    # Test probability predictions for specific class indices
    class_index = 1
    clf.predict_proba_class(X_test, class_index)
    class_index2 = 2
    clf.predict_proba_class(X_test, class_index2)

    # Plot the decision regions and data points
    plot_decision_regions(X=X_train, y=y_train, classifier=clf)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
