import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions


class Perceptron(object):
    """
    Implementation of the Perceptron algorithm for binary classification.
    """

    def __init__(self, eta=0.01, n_iter=10):
        # Learning rate (eta) controls the step size in updating the weights
        self.eta = eta
        # Number of iterations for training the perceptron
        self.n_iter = n_iter

    def fit(self, X, y):
        # Initialize weights with zeros (add one for the bias term)
        self.w_ = np.zeros(1 + X.shape[1])

        # Iterate over the training data for a fixed number of epochs
        for _ in range(self.n_iter):
            # Iterate over each training sample and its target label
            for xi, target in zip(X, y):
                # Compute the update value based on the difference between the predicted and target value
                update = self.eta * (target - self.predict(xi))
                # Update the weights: increment the weights by the update value times the input feature values
                self.w_[1:] += update * xi
                # Update the bias weight by the update value
                self.w_[0] += update

        return self

    def net_input(self, X):
        """
        Compute the net input (weighted sum of inputs and weights).
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Predict the class label for the input features.
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class Classifier:
    """
    Classifier that combines two perceptrons for multiclass classification.
    """

    def __init__(self, ppn1, ppn2):
        self.ppn1 = ppn1
        self.ppn2 = ppn2

    def predict(self, x):
        """
        Predict the class labels for the input features.
        """
        # Class 0: ppn1 predicts 1 and ppn2 predicts -1
        # Class 1: ppn1 predicts -1 and ppn2 predicts 1
        # Class 2: ppn1 and ppn2 predict -1
        return np.where(self.ppn1.predict(x) == 1, 0, np.where(self.ppn2.predict(x) == 1, 2, 1))


def main():
    # Load the iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]    # Select only two features: petal length and petal width
    y = iris.target
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create binary classification problems by selecting two classes at a time
    y0 = y_train.copy()  # Classify class 0 versus class 1
    y1 = y_train.copy()  # Classify class 1 versus class 2
    y2 = y_train.copy()  # Classify class 2 versus class 0
    X_sub = X_train.copy()

    # Map the class labels to -1 and 1 to create binary classification problems
    y0[(y_train == 1) | (y_train == 2)] = -1
    y0[(y0 == 0)] = 1

    y1[(y_train == 1) | (y_train == 0)] = -1
    y1[(y1 == 2)] = 1

    y2[(y_train == 2) | (y_train == 0)] = -1
    y2[(y2 == 1)] = 1

    print('y_train_01_subset: ', y0)
    print('y_train_03_subset: ', y1)

    # Train the first perceptron to classify class 0 versus class 1
    ppn1 = Perceptron(eta=0.1, n_iter=300)
    ppn1.fit(X_sub, y0)

    # Train the second perceptron to classify class 1 versus class 2
    ppn2 = Perceptron(eta=0.1, n_iter=300)
    ppn2.fit(X_sub, y1)

    # Train the third perceptron to classify class 2 versus class 0
    ppn3 = Perceptron(eta=0.1, n_iter=300)
    ppn3.fit(X_sub, y2)

    # Combine the perceptrons into a classifier
    clf = Classifier(ppn1, ppn2)

    # Plot the decision regions
    plot_decision_regions(X=X_train, y=y_train, classifier=clf)
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()