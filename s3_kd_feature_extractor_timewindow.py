"""
Keystroke Dynamics Feature Extractor

by Roberto Dillon, (c) 2025

This script processes CSV keystroke files to extract typing dynamics features in sliding time windows.
It analyzes keystroke data and generates features related to typing patterns that can be used
for machine learning analysis and user behavior modeling.

Features extracted per time window:
1. Average dwell time (key press duration)
2. Standard deviation of dwell time
3. Average flight time (time between key release and next key press)
4. Standard deviation of flight time
5. Error rate (number of backspaces / total keypresses) * 100 so it is expressed in percentage

Time windows:
- 5-second windows (5000ms)
- 1-second sliding hop (1000ms)
- Windows with no activity for 3+ seconds are discarded
"""

import os
import csv
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# Constants
WINDOW_SIZE_MS = 5000  # 5 seconds per analysis window
WINDOW_HOP_MS = 1000   # Slide by 1 second each time
INACTIVITY_THRESHOLD_MS = 3000  # 3 seconds of inactivity threshold
BACKSPACE_KEY = "[BACKSPACE]"

def load_keystroke_data(file_path: str) -> pd.DataFrame:
    """
    Load keystroke data from CSV file and convert to DataFrame.
    
    Parameters:
        file_path (str): Path to the keystroke CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the keystroke data
    """
    try:
        # Read CSV file with the correct column names
        df = pd.read_csv(file_path)
        
        # Ensure we have the required columns
        required_columns = ["timestamp_ms", "key", "action"]
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in {file_path}")
                return pd.DataFrame()
        
        # Ensure timestamp_ms column is numeric
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"])
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values(by="timestamp_ms")
        
        return df
    except Exception as e:
        print(f"Error loading keystroke data from {file_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def pair_key_events(df: pd.DataFrame) -> List[Dict]:
    """
    Pair press and release events for each key.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing keystroke events
        
    Returns:
        List[Dict]: List of keystroke pairs (press-release events)
    """
    key_events = []
    press_events = {}
    
    try:
        for _, row in df.iterrows():
            key = row["key"]
            action = row["action"]
            time_ms = row["timestamp_ms"]
            
            if action == "press":
                # Store the press event
                press_events[key] = {"key": key, "press_time": time_ms}
            elif action == "release":
                # If we have a matching press event, create a complete keystroke
                if key in press_events:
                    keystroke = press_events[key].copy()
                    keystroke["release_time"] = time_ms
                    keystroke["dwell_time"] = time_ms - keystroke["press_time"]
                    key_events.append(keystroke)
                    del press_events[key]
    except Exception as e:
        print(f"Error pairing key events: {e}")
    
    return key_events

def calculate_flight_times(keystrokes: List[Dict]) -> List[Dict]:
    """
    Calculate flight time between consecutive keystrokes.
    Flight time is defined as the time between a key release and the next key press.
    
    Parameters:
        keystrokes (List[Dict]): List of paired keystroke events
        
    Returns:
        List[Dict]: Keystrokes with flight times added
    """
    try:
        # Sort keystrokes by press time
        keystrokes = sorted(keystrokes, key=lambda k: k["press_time"])
        
        # Calculate flight times
        for i in range(1, len(keystrokes)):
            prev_release = keystrokes[i-1]["release_time"]
            current_press = keystrokes[i]["press_time"]
            keystrokes[i]["flight_time"] = current_press - prev_release
    except Exception as e:
        print(f"Error calculating flight times: {e}")
    
    return keystrokes

def extract_window_features(
    keystrokes: List[Dict], 
    window_start: float,
    window_end: float
) -> Optional[Dict]:
    """
    Extract typing dynamics features for a specific time window.
    
    Parameters:
        keystrokes (List[Dict]): List of keystroke events with timing data
        window_start (float): Start time of the window in ms
        window_end (float): End time of the window in ms
        
    Returns:
        Optional[Dict]: Dictionary of features or None if window should be discarded
    """
    try:
        # Filter keystrokes that fall within the window
        window_keystrokes = [
            k for k in keystrokes 
            if window_start <= k["press_time"] < window_end and 
               k["release_time"] <= window_end
        ]
        
        # Check if window should be discarded
        if not window_keystrokes:
            return None
            
        # Check for inactivity
        times = sorted([k["press_time"] for k in window_keystrokes] + 
                       [k["release_time"] for k in window_keystrokes])
        
        # Calculate maximum gap between consecutive events
        max_gap = 0
        for i in range(1, len(times)):
            gap = times[i] - times[i-1]
            max_gap = max(max_gap, gap)
            
        # Discard window if there's a gap > INACTIVITY_THRESHOLD_MS
        if max_gap > INACTIVITY_THRESHOLD_MS:
            return None
            
        # Extract dwell times
        dwell_times = [k["dwell_time"] for k in window_keystrokes]
        
        # Extract flight times (skip first keystroke as it has no flight time)
        flight_times = [
            k["flight_time"] for k in window_keystrokes 
            if "flight_time" in k
        ]
        
        # Count backspace keys
        backspace_count = sum(1 for k in window_keystrokes if k["key"] == BACKSPACE_KEY)
        total_keys = len(window_keystrokes)
        # error rate ratio * 100
        error_rate = (backspace_count / total_keys)*100 if total_keys > 0 else 0
        
        # Calculate statistics
        features = {
            "window_start_ms": window_start,
            "window_end_ms": window_end,
            "avg_dwell_time": np.mean(dwell_times) if dwell_times else 0,
            "std_dwell_time": np.std(dwell_times) if len(dwell_times) > 1 else 0,
            "avg_flight_time": np.mean(flight_times) if flight_times else 0,
            "std_flight_time": np.std(flight_times) if len(flight_times) > 1 else 0,
            "error_rate": error_rate,
            "key_count": total_keys
        }
        
        return features
    except Exception as e:
        print(f"Error extracting window features: {e}")
        return None

def process_file(input_file: str, output_file: str) -> None:
    """
    Process a keystroke file and generate a feature file.
    
    Parameters:
        input_file (str): Path to input CSV file
        output_file (str): Path to output feature CSV file
    """
    try:
        print(f"Processing {input_file}...")
        
        # Load data
        df = load_keystroke_data(input_file)
        if df.empty:
            print(f"Skipping empty file: {input_file}")
            return
            
        # Get time range
        min_time = df["timestamp_ms"].min()
        max_time = df["timestamp_ms"].max()
        
        # Pair press and release events
        keystroke_pairs = pair_key_events(df)
        
        # Calculate flight times
        keystroke_pairs = calculate_flight_times(keystroke_pairs)
        
        # Define sliding windows
        window_starts = np.arange(min_time, max_time, WINDOW_HOP_MS)
        
        # Extract features for each window
        all_features = []
        for start in window_starts:
            end = start + WINDOW_SIZE_MS
            features = extract_window_features(keystroke_pairs, start, end)
            if features:
                all_features.append(features)
        
        # Save features to CSV
        if all_features:
            features_df = pd.DataFrame(all_features)
            # Only keep the required feature columns
            feature_columns = [
                "window_start_ms", "window_end_ms", 
                "avg_dwell_time", "std_dwell_time", 
                "avg_flight_time", "std_flight_time", 
                "error_rate"
            ]
            features_df = features_df[feature_columns]
            features_df.to_csv(output_file, index=False)
            print(f"Feature file saved: {output_file}")
        else:
            print(f"No valid features extracted from {input_file}")
            
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

def main():
    """
    Main function to process all keystroke CSV files in the current directory.
    """
    try:
        # Find all CSV files in the current directory
        input_files = glob.glob("*.csv")
        
        # Filter files that appear to be keystroke files (not feature files)
        # Now checking for the required columns in the header
        keystroke_files = []
        for file in input_files:
            try:
                with open(file, 'r') as f:
                    header = f.readline().strip().lower()
                    if all(col in header for col in ["timestamp_ms", "key", "action"]):
                        keystroke_files.append(file)
            except Exception as e:
                print(f"Error reading header from {file}: {e}")
                
        print(f"Found {len(keystroke_files)} keystroke files to process")
        
        # Process each file
        for input_file in keystroke_files:
            # Generate output filename
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_features.csv"
            
            # Process the file
            process_file(input_file, output_file)
            
        print("Processing complete")
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()