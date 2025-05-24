# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:49:46 2025

by Roberto Dillon, (c) 2025

Keystroke Feature Preprocessing Script

This script preprocesses keystroke feature data by:
1. Standardizing features (removing mean and scaling to unit variance)
2. Performing Principal Component Analysis (PCA) to reduce dimensions from 5 to 2

For each input feature CSV file, the script creates two output files:
- A standardized version of the original data
- A PCA-reduced version with just 2 dimensions

The script processes all feature CSV files in the current directory, identified by
the standard columns from our feature extraction process.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

# Expected feature columns from the feature extraction output
FEATURE_COLUMNS = [
    "avg_dwell_time", 
    "std_dwell_time", 
    "avg_flight_time", 
    "std_flight_time", 
    "error_rate"
]

def is_feature_file(file_path: str) -> bool:
    """
    Check if a file is a keystroke feature file based on its header.
    
    Parameters:
        file_path (str): Path to the CSV file to check
        
    Returns:
        bool: True if the file appears to be a feature file, False otherwise
    """
    try:
        df = pd.read_csv(file_path, nrows=1)
        # Check if all expected feature columns are present
        return all(col in df.columns for col in FEATURE_COLUMNS)
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return False

def load_feature_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load feature data from a CSV file.
    
    Parameters:
        file_path (str): Path to the feature CSV file
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with feature data or None if loading fails
    """
    try:
        df = pd.read_csv(file_path)
        
        # Ensure all required columns are present
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in {file_path}: {missing_cols}")
            return None
            
        return df
    except Exception as e:
        print(f"Error loading feature data from {file_path}: {e}")
        return None

def standardize_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Parameters:
        df (pd.DataFrame): DataFrame with feature data
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Standardized data and fitted scaler
    """
    try:
        # Create a copy of the original DataFrame
        result_df = df.copy()
        
        # Create the scaler
        scaler = StandardScaler()
        
        # Fit and transform the feature columns
        scaled_features = scaler.fit_transform(df[FEATURE_COLUMNS])
        
        # Replace the original feature columns with scaled values
        for i, col in enumerate(FEATURE_COLUMNS):
            result_df[col] = scaled_features[:, i]
            
        return result_df, scaler
    except Exception as e:
        print(f"Error standardizing features: {e}")
        # Return original data and None scaler on error
        return df, None

def apply_pca(df: pd.DataFrame, n_components: int = 2) -> Tuple[pd.DataFrame, PCA]:
    """
    Apply Principal Component Analysis for dimensionality reduction.
    
    Parameters:
        df (pd.DataFrame): DataFrame with standardized feature data
        n_components (int): Number of components to keep (default: 2)
        
    Returns:
        Tuple[pd.DataFrame, PCA]: DataFrame with PCA components and fitted PCA object
    """
    try:
        # Extract feature data for PCA
        X = df[FEATURE_COLUMNS].values
        
        # Initialize PCA
        pca = PCA(n_components=n_components)
        
        # Fit and transform the data
        pca_components = pca.fit_transform(X)
        
        # Create a new DataFrame with the principal components
        pca_df = pd.DataFrame(
            data=pca_components, 
            columns=[f'principal_component_{i+1}' for i in range(n_components)]
        )
        
        # Add window start/end times if they exist in the original DataFrame
        if 'window_start_ms' in df.columns:
            pca_df['window_start_ms'] = df['window_start_ms'].values
        if 'window_end_ms' in df.columns:
            pca_df['window_end_ms'] = df['window_end_ms'].values
            
        # Calculate and print explained variance
        explained_variance = pca.explained_variance_ratio_.sum() * 100
        print(f"PCA explained variance: {explained_variance:.2f}%")
        
        return pca_df, pca
    except Exception as e:
        print(f"Error applying PCA: {e}")
        # Return empty DataFrame and None PCA object on error
        return pd.DataFrame(), None

def process_file(input_file: str) -> None:
    """
    Process a feature file, standardize data, and apply PCA.
    
    Parameters:
        input_file (str): Path to input feature CSV file
    """
    try:
        print(f"Processing {input_file}...")
        
        # Generate output filenames
        base_name = os.path.splitext(input_file)[0]
        standardized_file = f"{base_name}_standardized.csv"
        pca_file = f"{base_name}_pca.csv"
        
        # Load data
        df = load_feature_data(input_file)
        if df is None or df.empty:
            print(f"Skipping {input_file}: No valid data found")
            return
            
        # Standardize features
        standardized_df, scaler = standardize_features(df)
        if scaler is None:
            print(f"Warning: Feature standardization failed for {input_file}")
            return
            
        # Save standardized data
        standardized_df.to_csv(standardized_file, index=False)
        print(f"Standardized data saved to {standardized_file}")
        
        # Apply PCA
        pca_df, pca = apply_pca(standardized_df)
        if pca is None or pca_df.empty:
            print(f"Warning: PCA failed for {input_file}")
            return
            
        # Save PCA data
        pca_df.to_csv(pca_file, index=False)
        print(f"PCA data saved to {pca_file}")
        
        # Optional: Create a visualization of feature importance
        try:
            feature_importance = np.abs(pca.components_)
            feature_importance = feature_importance / feature_importance.sum(axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.bar(FEATURE_COLUMNS, feature_importance[0], alpha=0.7, label='PC1')
            plt.bar(FEATURE_COLUMNS, feature_importance[1], alpha=0.5, label='PC2')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'Feature Importance in PCA - {os.path.basename(input_file)}')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            plot_file = f"{base_name}_feature_importance.png"
            plt.savefig(plot_file)
            plt.close()
            print(f"Feature importance plot saved to {plot_file}")
        except Exception as e:
            print(f"Warning: Failed to create feature importance plot: {e}")
        
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

def main():
    """
    Main function to process all feature CSV files in the current directory.
    """
    try:
        # Find all CSV files in the current directory
        input_files = glob.glob("*.csv")
        
        # Filter files that appear to be feature files
        feature_files = [file for file in input_files if is_feature_file(file)]
        
        print(f"Found {len(feature_files)} feature files to process")
        
        # Process each file
        for input_file in feature_files:
            process_file(input_file)
            
        print("Processing complete")
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()