'''
Project eStyler - Free-text Keyboard Dynamics
by Roberto Dillon - roberto.dillon@ieee.org

user_data_generator.py : 
This file generates data by simulating users with different typing speeds for testing purposes

Make sure you have all the relevant libraries installed:
e.g. 
pip install pynput pandas

Usage:
Hardcode desired user's values in the line at the end of the script
e.g.
file_path = generate_typing_data(60, 0.7, 150) #generates data for a 150-words text by a user typing at 60 wps and 70% accuracy.

then call the script:
python keyStyler.py 


'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_typing_data(wpm=40, accuracy=0.90, num_words=150, window_size=5, hop_size=1):
    #start_time = datetime(2024, 6, 8, 10, 0, 0)
    start_time = datetime.now()
    data = []

    # Calculate typing duration based on words per minute
    typing_duration = int((num_words / wpm) * 60)  # seconds for the given words per minute
    total_intervals = typing_duration  # 1-second intervals

    # Calculate error intervals based on accuracy
    error_intervals = [i for i in range(0, total_intervals, int(total_intervals / (num_words * (1 - accuracy))))]  # Errors distributed to reflect accuracy

    # Average times for 60 WPM
    base_dwell_time = 0.060  # 60 ms
    base_flight_time = 0.107  # 107 ms

    # Adjust times based on WPM
    scale_factor = 60 / wpm
    dwell_time_mean = base_dwell_time * scale_factor
    flight_time_mean = base_flight_time * scale_factor

    # Randomizing avg_dwell_time and avg_flight_time based on wpm
    np.random.seed(42)  # For reproducibility

    # Generate key press data for each second
    key_press_data = []
    for i in range(total_intervals):
        avg_dwell_time = round(np.random.normal(dwell_time_mean, 0.005), 5)
        avg_flight_time = round(np.random.normal(flight_time_mean, 0.05), 5)
        error = 1 - accuracy if i in error_intervals else 0.0
        key_press_data.append((avg_dwell_time, avg_flight_time, error))
    
    print(f"Generated key press data for {total_intervals} intervals")

    # Process data using a sliding window
    for i in range(0, total_intervals - window_size + 1, hop_size):
        window_data = key_press_data[i:i + window_size]
        if len(window_data) < window_size:
            break

        avg_dwell_time = round(np.mean([d[0] for d in window_data]), 5)
        avg_flight_time = round(np.mean([d[1] for d in window_data]), 5)
        error_ratio = round(np.mean([d[2] for d in window_data]), 5)
        
        timestamp = start_time + timedelta(seconds=i)
        data.append({
            'timestamp': timestamp.isoformat(),
            'avg_dwell_time': avg_dwell_time,
            'avg_flight_time': avg_flight_time,
            'error_ratio': error_ratio
        })

    print(f"Processed data into windows. Total windows: {len(data)}")

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Print the DataFrame to debug
    print(df.head())

    # Save the DataFrame to a CSV file
    file_path = 'keyboard_dynamics_simulated_wpm{}_acc{}_words{}.csv'.format(wpm, accuracy, num_words)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    return file_path

# Generate data for a user with wpm=60, accuracy=0.7, words=150
file_path = generate_typing_data(60, 0.7, 150)
print(file_path)
