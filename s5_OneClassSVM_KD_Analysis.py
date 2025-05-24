# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:03:16 2025

by Roberto Dillon, (c) 2025


OneClass SVM for Keystroke Dynamics Analysis

This script implements One-Class SVM to analyze keystroke dynamics data that has been
preprocessed through PCA. It detects whether test samples match the typing pattern
of the training data or represent anomalous/different typing behavior.

Features:
- Loads PCA-reduced keystroke dynamics data (2D)
- Trains a One-Class SVM model on a training file
- Tests the model on a separate test file
- Visualizes the decision boundary and classification results
- Reports classification statistics (match/anomaly percentages)

Usage:
1. Set the file_training and file_test variables to point to your PCA datasets
2. Adjust the SVM parameters as needed for your specific use case
3. Run the script to see classification results and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from typing import Tuple, List, Dict, Any
import os
import time

# ================= CONFIGURATION =================
# User-configurable variables - MODIFY THESE VALUES as needed

# Input file paths (modify these with your actual file paths)
file_training = "user1_features_pca.csv"  # PCA file for training
file_test = "attacker_features_pca.csv"  # PCA file for testing

# One-Class SVM parameters (modify these to experiment with different settings)
svm_params = {
    "kernel": "rbf",     # Kernel type: 'rbf', 'linear', 'poly', 'sigmoid'
    "gamma": 0.2,        # Kernel coefficient for rbf, poly and sigmoid kernels
    "nu": 0.01,          # Upper bound on the fraction of training errors (0.0-1.0)
    "degree": 3,         # Degree of polynomial kernel (ignored for other kernels)
}

# Visualization settings
plot_density = 100  # Higher values create smoother decision boundaries but slower rendering

# ================= CODE IMPLEMENTATION =================

def load_pca_data(file_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load PCA-reduced keystroke dynamics data.
    
    Parameters:
        file_path (str): Path to the PCA CSV file
        
    Returns:
        Tuple[np.ndarray, pd.DataFrame]: X features array and full DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check for principal component columns
        pc_columns = [col for col in df.columns if col.startswith('principal_component_')]
        if len(pc_columns) < 2:
            raise ValueError(f"File does not contain expected PCA columns: {file_path}")
        
        # Extract features (first two principal components)
        X = df[pc_columns[:2]].values
        
        return X, df
    except Exception as e:
        print(f"Error loading PCA data from {file_path}: {e}")
        raise

def train_oneclass_svm(X_train: np.ndarray, params: Dict[str, Any]) -> OneClassSVM:
    """
    Train a One-Class SVM model.
    
    Parameters:
        X_train (np.ndarray): Training feature array
        params (Dict[str, Any]): SVM parameters
        
    Returns:
        OneClassSVM: Trained model
    """
    print(f"Training One-Class SVM with parameters: {params}")
    
    try:
        # Create and fit the model
        model = OneClassSVM(**params)
        start_time = time.time()
        model.fit(X_train)
        training_time = time.time() - start_time
        
        print(f"Model training completed in {training_time:.3f} seconds")
        return model
    except Exception as e:
        print(f"Error training One-Class SVM: {e}")
        raise

def predict_and_evaluate(model: OneClassSVM, X_test: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Make predictions and calculate evaluation metrics.
    
    Parameters:
        model (OneClassSVM): Trained One-Class SVM model
        X_test (np.ndarray): Test feature array
        
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Predictions and evaluation metrics
    """
    try:
        # Make predictions (-1 for outliers, 1 for inliers)
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate statistics
        n_samples = len(y_pred)
        n_inliers = np.sum(y_pred == 1)
        n_outliers = np.sum(y_pred == -1)
        
        inlier_percent = n_inliers / n_samples * 100
        outlier_percent = n_outliers / n_samples * 100
        
        metrics = {
            "n_samples": n_samples,
            "n_inliers": n_inliers,
            "n_outliers": n_outliers,
            "inlier_percent": inlier_percent,
            "outlier_percent": outlier_percent,
            "prediction_time": prediction_time
        }
        
        return y_pred, metrics
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise

def plot_results(
    model: OneClassSVM, 
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_pred: np.ndarray,
    training_file: str,
    test_file: str,
    plot_density: int = 100
) -> None:
    """
    Visualize the results with decision boundary.
    
    Parameters:
        model (OneClassSVM): Trained One-Class SVM model
        X_train (np.ndarray): Training feature array
        X_test (np.ndarray): Test feature array
        y_pred (np.ndarray): Prediction results
        training_file (str): Training file name for plot title
        test_file (str): Test file name for plot title
        plot_density (int): Density of the mesh grid for decision boundary
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Determine plot boundaries
        all_data = np.vstack((X_train, X_test))
        x_min, x_max = all_data[:, 0].min() - 1, all_data[:, 0].max() + 1
        y_min, y_max = all_data[:, 1].min() - 1, all_data[:, 1].max() + 1
        
        # Create a mesh grid for the decision boundary
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, plot_density),
            np.linspace(y_min, y_max, plot_density)
        )
        
        # Get the decision function values
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r, alpha=0.5)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        
        # Plot training data
        plt.scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolors='k', 
                    s=80, marker='o', label='Training Data')
        
        # Plot test data points colored by prediction
        inliers = y_pred == 1
        outliers = y_pred == -1
        
        plt.scatter(X_test[inliers, 0], X_test[inliers, 1], c='green', 
                    s=60, marker='^', label='Test Data (Matches Training)')
        plt.scatter(X_test[outliers, 0], X_test[outliers, 1], c='red', 
                    s=60, marker='x', label='Test Data (Anomalous)')
        
        # Add labels and legend
        plt.title(f'One-Class SVM Analysis\nTraining: {os.path.basename(training_file)} | Test: {os.path.basename(test_file)}', fontsize=14)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the plot
        base_name = os.path.splitext(os.path.basename(test_file))[0]
        plot_file = f"{base_name}_oneclass_svm_results.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        
        print(f"Results plot saved as: {plot_file}")
    except Exception as e:
        print(f"Error plotting results: {e}")

def main():
    """
    Main function to run the One-Class SVM analysis.
    """
    try:
        print("=" * 60)
        print("One-Class SVM for Keystroke Dynamics Analysis")
        print("=" * 60)
        
        # Check if files exist
        if not os.path.exists(file_training):
            raise FileNotFoundError(f"Training file not found: {file_training}")
        if not os.path.exists(file_test):
            raise FileNotFoundError(f"Test file not found: {file_test}")
        
        # Load data
        print(f"Loading training data from: {file_training}")
        X_train, df_train = load_pca_data(file_training)
        
        print(f"Loading test data from: {file_test}")
        X_test, df_test = load_pca_data(file_test)
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        # Train the model
        model = train_oneclass_svm(X_train, svm_params)
        
        # Make predictions
        y_pred, metrics = predict_and_evaluate(model, X_test)
        
        # Display results
        print("\nClassification Results:")
        print(f"Total test samples: {metrics['n_samples']}")
        print(f"Classified as matching training pattern: {metrics['n_inliers']} ({metrics['inlier_percent']:.2f}%)")
        print(f"Classified as different/anomalous: {metrics['n_outliers']} ({metrics['outlier_percent']:.2f}%)")
        print(f"Prediction time: {metrics['prediction_time']:.3f} seconds")
        
        # Plot results
        plot_results(model, X_train, X_test, y_pred, file_training, file_test, plot_density)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()