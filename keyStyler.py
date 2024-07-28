'''
Project eStyler - Free-text Keyboard Dynamics
by Roberto Dillon - roberto.dillon@ieee.org


keyStyler.py : keylogger for freetext keyboard dynamics

1. runs in the background till the user presses the Esc key and, once done, saves all data in a .csv file, named after the current timestamp
2. computes the following parameters: average dwell time (the time a specific key is pressed) and flight time (time between two key presses)
 for all keys pressed during a shifting time window where width and hop value can be set by the user as input parameters when calling the script
 (default values: width = 5 seconds, hop = 1 second)
3. ignores a key press if this is a Shift, Alt or Ctrl key
4. includes also a parameter called "error ratio" per time window defined as the number of times any one key between the Del, Back and cursor
 (left, right, up or down keys) are pressed.
5. ignores extracted data if the time window under analysis has no key presses for a time greater than 3 seconds
 (in other words, the user takes a pause from typing) but starts again with a new window as soon as keypresses happen again.

Make sure you have all the relevant libraries installed:
e.g. 
pip install pynput pandas

Usage:
python keyStyler.py --width 5 --hop 1

Help Function: The script includes a help function using argparse to explain how to call it with parameters.
You can run the script from the command line and adjust the width and hop parameters as needed.

'''

import pandas as pd
import time
from datetime import datetime
from pynput import keyboard
import threading
import argparse

# Constants for ignored keys
IGNORED_KEYS = {keyboard.Key.shift, keyboard.Key.shift_r, keyboard.Key.alt, keyboard.Key.alt_r, keyboard.Key.ctrl, keyboard.Key.ctrl_r}
ERROR_KEYS = {keyboard.Key.backspace, keyboard.Key.delete, keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down}

# Global variables
key_events = []
stop_event = threading.Event()
last_event_time = None

# Function to compute metrics
def compute_metrics(window_events):
    try:
        if not window_events:
            return None

        dwell_times = []
        flight_times = []
        error_count = 0

        press_time = None

        for i, event in enumerate(window_events):
            if event['key'] in ERROR_KEYS:
                error_count += 1
            if event['event'] == 'press':
                if press_time is not None:
                    flight_times.append(event['time'] - press_time)
                press_time = event['time']
            elif event['event'] == 'release':
                if press_time is not None:
                    dwell_times.append(event['time'] - press_time)
        
        avg_dwell_time = sum(dwell_times) / len(dwell_times) if dwell_times else 0
        avg_flight_time = sum(flight_times) / len(flight_times) if flight_times else 0

        return avg_dwell_time, avg_flight_time, error_count / len(window_events)
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None

# Listener callback functions
def on_press(key):
    global last_event_time
    try:
        if key in IGNORED_KEYS:
            return
        current_time = time.time()
        key_events.append({'key': key, 'event': 'press', 'time': current_time})
        last_event_time = current_time
    except Exception as e:
        print(f"Error on key press: {e}")

def on_release(key):
    global last_event_time
    try:
        if key in IGNORED_KEYS:
            return
        current_time = time.time()
        key_events.append({'key': key, 'event': 'release', 'time': current_time})
        last_event_time = current_time
        if key == keyboard.Key.esc:
            stop_event.set()
            return False
    except Exception as e:
        print(f"Error on key release: {e}")

# Background listener thread
def start_listener():
    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except Exception as e:
        print(f"Error in listener thread: {e}")

# Main function to run the analysis
def main(width, hop):
    global last_event_time
    try:
        listener_thread = threading.Thread(target=start_listener)
        listener_thread.start()

        results = []
        #start_time = time.time()
        while not stop_event.is_set():
            current_time = time.time()
            window_start = current_time - width
            window_events = [event for event in key_events if window_start <= event['time'] <= current_time]
            if window_events and (current_time - last_event_time <= 3):
                metrics = compute_metrics(window_events)
                if metrics:
                    avg_dwell_time, avg_flight_time, error_ratio = metrics
                    results.append({'timestamp': datetime.now().isoformat(), 'avg_dwell_time': avg_dwell_time, 'avg_flight_time': avg_flight_time, 'error_ratio': error_ratio})
            time.sleep(hop)
        
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df = pd.DataFrame(results)
            df.to_csv(f'keyboard_dynamics_{timestamp}.csv', index=False)
            print("Results saved to .csv file. Game Over. Do you want to play again?")
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Keyboard dynamics analyzer.')
        parser.add_argument('--width', type=int, default=5, help='Width of the time window in seconds.')
        parser.add_argument('--hop', type=int, default=1, help='Hop value in seconds.')
        args = parser.parse_args()
        
        print("keyStyler v.0.0.2 - Keyboard Dynamics Extractor (c) Roberto Dillon 2024")
        print("Press ESC to stop the recording.")
        main(args.width, args.hop)
    except Exception as e:
        print(f"Error in script initialization: {e}")



'''


'''
