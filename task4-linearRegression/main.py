import numpy as np
from sklearn.model_selection import train_test_split

# Load data from file
data = np.loadtxt("dane/dane16.txt")

# Split data into input and output variables
X = data[:, 0]
y = data[:, 1]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear regression
# Add a column of ones for w0
X_train_m1 = np.c_[np.ones(X_train.shape[0]), X_train]

# Compute model parameters using the least squares method
w_m1 = np.linalg.inv(X_train_m1.T.dot(X_train_m1)).dot(X_train_m1.T).dot(y_train)
w0_m1, w1_m1 = w_m1

# Verify quality of Model 1
# Add a column of ones for w0
X_test_m1 = np.c_[np.ones(X_test.shape[0]), X_test]

# Predict output variable for test set
y_pred_m1 = X_test_m1.dot(w_m1)

# Compute mean squared error
mse_m1 = np.mean((y_test - y_pred_m1)**2)
print("MSE for Model 1:", mse_m1)

# Model 2: Quadratic regression
# Add columns for w0, w1, and w2
X_train_m2 = np.c_[np.ones(X_train.shape[0]), X_train, X_train**2]

# Compute model parameters using the least squares method
w_m2 = np.linalg.inv(X_train_m2.T.dot(X_train_m2)).dot(X_train_m2.T).dot(y_train)
w0_m2, w1_m2, w2_m2 = w_m2

# Verify quality of Model 2
# Add columns for w0, w1, and w2
X_test_m2 = np.c_[np.ones(X_test.shape[0]), X_test, X_test**2]

# Predict output variable for test set
y_pred_m2 = X_test_m2.dot(w_m2)

# Compute mean squared error
mse_m2 = np.mean((y_test - y_pred_m2)**2)
print("MSE for Model 2:", mse_m2)

# Compare both models
if mse_m1 < mse_m2:
    print("Model 1 is better than Model 2.")
else:
    print("Model 2 is better than Model 1.")
