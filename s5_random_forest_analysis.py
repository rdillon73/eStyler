# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 11:26:06 2025

by Roberto Dillon, (c) 2025

This script determines whether a test file belongs to the same distribution as a training file using a Random Forest classifier approach. Here's how it works:
Key Components of the Script:

Data Loading and Preprocessing:

Loads CSV files for both training and test data
Extracts features (all columns except the window timestamps)


Distribution Comparison Approach:

Labels training data as class 0 and test data as class 1
Trains a Random Forest to distinguish between the two sets
If the model struggles to tell them apart (accuracy close to 50%), the datasets likely come from the same distribution
If the model can easily distinguish them (high accuracy), they likely come from different distributions


Additional Validation:

Performs Kolmogorov-Smirnov tests on each feature to corroborate the Random Forest findings
Calculates feature importance to identify which features differ most between datasets


Visualization and Results:

Feature importance plot
KS test p-values
Distribution comparison of the most important feature
Clear conclusion about whether datasets appear to be from the same distribution



How to Use the Script:

Set the File Paths:

train_file = "path/to/your/training_file.csv"
test_file = "path/to/your/test_file.csv"

Customize Random Forest Parameters:
    
rf_params = {
    'n_estimators': 200,        # Increase for better performance
    'max_depth': 10,            # Limit tree depth to prevent overfitting
    'min_samples_split': 5,     # More samples required to split nodes
    # Add other parameters as needed
}

Interpreting Results:

Accuracy below 70% suggests datasets are from the same distribution
KS test p-values > 0.05 suggest same distribution for individual features
The visualization helps identify specific differences between datasets

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Extract features and target"""
    # Using all columns except the first two (window timestamps) as features
    features = df.iloc[:, 2:].values
    return features

def train_random_forest(train_file, test_file, rf_params=None):
    """
    Train a Random Forest model to distinguish between training and test distributions
    
    Parameters:
    -----------
    train_file : str
        Path to the training CSV file
    test_file : str
        Path to the test CSV file
    rf_params : dict, optional
        Parameters for RandomForestClassifier
        
    Returns:
    --------
    dict
        Results of the analysis
    """
    # Default Random Forest parameters
    if rf_params is None:
        rf_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto',
            'bootstrap': True,
            'random_state': 42
        }
    
    # Load datasets
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    
    # Preprocess data
    train_features = preprocess_data(train_data)
    test_features = preprocess_data(test_data)
    
    # Create labels: 0 for training data, 1 for test data
    train_labels = np.zeros(train_features.shape[0])
    test_labels = np.ones(test_features.shape[0])
    
    # Combine data
    all_features = np.vstack((train_features, test_features))
    all_labels = np.concatenate((train_labels, test_labels))
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        all_features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_val)
    
    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    # Calculate feature importance
    feature_names = train_data.columns[2:]
    feature_importance = dict(zip(feature_names, rf.feature_importances_))
    
    # Perform KS test for each feature to corroborate Random Forest findings
    ks_test_results = {}
    for i, feature_name in enumerate(feature_names):
        statistic, p_value = ks_2samp(train_features[:, i], test_features[:, i])
        ks_test_results[feature_name] = {
            'statistic': statistic,
            'p_value': p_value,
            'same_distribution': p_value > 0.05
        }
    
    # Determine if distributions are similar
    # If accuracy is close to 0.5, the model struggles to distinguish between distributions
    # which suggests they are similar
    is_same_distribution = accuracy < 0.7  # original: 0.7
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'feature_importance': feature_importance,
        'ks_test_results': ks_test_results,
        'is_same_distribution': is_same_distribution,
        'model': rf,
        'data': {
            'train_features': train_features,
            'test_features': test_features
        }
    }

def plot_results(results):
    """Plot analysis results"""
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot feature importance
    importance_df = pd.DataFrame({
        'Feature': list(results['feature_importance'].keys()),
        'Importance': list(results['feature_importance'].values())
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=axs[0, 0])
    axs[0, 0].set_title('Feature Importance')
    axs[0, 0].set_xlabel('Importance')
    axs[0, 0].set_ylabel('Feature')
    
    # Plot confusion matrix
    cm = confusion_matrix(
        np.concatenate([np.zeros(len(results['data']['train_features'])), 
                       np.ones(len(results['data']['test_features']))]),
        np.concatenate([np.zeros(len(results['data']['train_features'])), 
                       np.ones(len(results['data']['test_features']))])
    )
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 1])
    axs[0, 1].set_title('Perfect Confusion Matrix (For Reference)')
    axs[0, 1].set_xlabel('Predicted')
    axs[0, 1].set_ylabel('Actual')
    
    # Plot KS test p-values
    ks_df = pd.DataFrame({
        'Feature': list(results['ks_test_results'].keys()),
        'p-value': [v['p_value'] for v in results['ks_test_results'].values()]
    }).sort_values('p-value')
    
    sns.barplot(x='p-value', y='Feature', data=ks_df, ax=axs[1, 0])
    axs[1, 0].axvline(x=0.05, color='r', linestyle='--')
    axs[1, 0].set_title('KS Test p-values (red line at p=0.05)')
    axs[1, 0].set_xlabel('p-value')
    axs[1, 0].set_ylabel('Feature')
    
    # Plot feature distribution comparison for most important feature
    most_important_feature = importance_df.iloc[0]['Feature']
    feature_idx = list(results['feature_importance'].keys()).index(most_important_feature)
    
    sns.histplot(results['data']['train_features'][:, feature_idx], 
                 color='blue', alpha=0.5, label='Training', ax=axs[1, 1])
    sns.histplot(results['data']['test_features'][:, feature_idx], 
                 color='red', alpha=0.5, label='Test', ax=axs[1, 1])
    
    axs[1, 1].set_title(f'Distribution of Most Important Feature: {most_important_feature}')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nRandom Forest Classification Report:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    for key, value in results['classification_report'].items():
        if key not in ('accuracy', 'macro avg', 'weighted avg'):
            print(f"Class {key}: Precision: {value['precision']:.4f}, Recall: {value['recall']:.4f}, F1: {value['f1-score']:.4f}")
    
    # Print conclusion
    print("\nConclusion:")
    if results['is_same_distribution']: 
        print("✅ The test file APPEARS to be from the same distribution as the training file.")
    else:
        print("❌ The test file APPEARS to be from a different distribution than the training file.")
    
    print("\nKS Test Results (p > 0.05 suggests same distribution):")
    for feature, result in results['ks_test_results'].items():
        status = "✓" if result['same_distribution'] else "✗" # {status}
        print(f" {feature}: p-value = {result['p_value']:.4f}")

def main():
    # Define input files (MODIFY THESE PATHS)
    train_file = "your_training_file.csv"  # Path to training file
    test_file = "your_test_file.csv"   # Path to test file (using same file for demo)
    
    # Define Random Forest parameters 
    rf_params = {
        'n_estimators': 500,      # 100 - Number of trees in the forest
        'max_depth': 10,        # None - Maximum depth of trees (None for unlimited)
        'min_samples_split': 5,   # 10 - 2 - Minimum samples required to split a node
        'min_samples_leaf':2,    # 5 - Minimum samples required at a leaf node
        'max_features': 'sqrt',   # auto, sqrt- Number of features to consider for best split
        'bootstrap': True,        # Whether to use bootstrap samples
        'random_state': 42,       # Random seed for reproducibility
        'class_weight': 'balanced' # Handle class imbalance
    }
    
    # Analyze data
    results = train_random_forest(train_file, test_file, rf_params)
    
    # Plot and print results
    plot_results(results)

if __name__ == "__main__":
    main()