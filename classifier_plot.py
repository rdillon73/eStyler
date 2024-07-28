'''
Project eStyler - Free-text Keyboard Dynamics
by Roberto Dillon - roberto.dillon@ieee.org


classifier_plot.py

Explanation:

This script implements a One-Class SVM (Support Vector Machine), a popular choice in ML for novelty detection. 
It tries to find a decision boundary that encloses the majority of the training data points and flags new points outside
this boundary as anomalies.

1. Data is scaled and PCA is to reduce the 3D feature space (average dwell time, average flight time, and error ratio) to 2D for visualization.
2. Classification is performed
3. The training and test data points are plotted in the reduced 2D PCA space.
4. The decision boundary of the One-Class SVM is plotted using a contour plot.

This approach ensures that the three-dimensional data is represented in a 2D scatter plot while maintaining the relationships between the features as much as possible.

The script provides a detailed summary of the prediction results, including the total number of predictions and the percentages of samples classified as "same user"
and "different user".


Make sure you have all the relevant libraries installed:
e.g. 
pip install numpy pandas

Usage:
Hardcode the training and test csv files in the 'train_data' and 'test_data' respectively
e.g.
train_data = pd.read_csv('legit_user.csv')
test_data = pd.read_csv('typing_user.csv') # specify path to testing file


Then call the script:
python classifier_plot.py 



'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load your training data
train_data = pd.read_csv('legit_user.csv')  # specify path to training file

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data)

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Train a One-Class SVM
#oc_svm = OneClassSVM(kernel="rbf", degree=2, gamma=0.2, nu=0.16).fit(X_train)
oc_svm = OneClassSVM(kernel="rbf", degree=2, gamma=0.2, nu=0.01).fit(X_train)


# Load new data to test
test_data = pd.read_csv('typing_user.csv') # specify path to testing file
X_test = scaler.transform(test_data)
X_test_pca = pca.transform(X_test)

# Predict whether new data points are from the same user
predictions = oc_svm.predict(X_test)

# Interpret predictions
# -1 indicates anomalies (not the same user), 1 indicates inliers (same user)
results = ['Same User' if x == 1 else 'Different User' for x in predictions]

# Print results
print(results)

# Summary
num_predictions = len(predictions)
num_same_user = results.count('Same User')
num_different_user = results.count('Different User')
percentage_same_user = (num_same_user / num_predictions) * 100
percentage_different_user = (num_different_user / num_predictions) * 100

print(f"\nSummary:")
print(f"Total number of predictions: {num_predictions}")
print(f"Number of 'Same User' predictions: {num_same_user} ({percentage_same_user:.2f}%)")
print(f"Number of 'Different User' predictions: {num_different_user} ({percentage_different_user:.2f}%)")

# Plot the training and test data in PCA-reduced 2D space
plt.figure(figsize=(10, 6))

# Scatter plot of the training data
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='blue', label='Training Data')

# Scatter plot of the test data
colors = ['red' if label == -1 else 'green' for label in predictions]
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=colors, label='Test Data')

# Plot the decision function
xx, yy = np.meshgrid(np.linspace(X_train_pca[:, 0].min(), X_train_pca[:, 0].max(), 500),
                     np.linspace(X_train_pca[:, 1].min(), X_train_pca[:, 1].max(), 500))
Z = oc_svm.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Contour plot of the decision function
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='dashed')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('One-Class SVM - User Verification')
plt.legend()
plt.show()
