# -*- coding: utf-8 -*-
'''
ABM Dataset Generator v1.0

by Roberto Dillon, (c) 2025

This script generates keystroke datasets for behavioral biometrics research using the
Agent-Based Modeling (ABM) typing simulator. It creates realistic typing samples for
both legitimate users and potential attackers, which can be used for:

1. Training user authentication systems
2. Testing impostor detection algorithms
3. Behavioral biometrics research

Dataset Structure:
- 5 Agents (3 legitimate users, 2 attackers)
- 4 files per agent (2 keyboard types × 2 sessions)
- Each file contains ~1000 characters of typing data

The legitimate users have consistent typing patterns within reasonable variation,
while attackers are modeled with distinctly different typing rhythms and behaviors.
'''

import csv
import os
import numpy as np
import random
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import the FreeTextTypingAgent class from the ABM module
try:
    from s1_ABM_typing_simulator_v2 import FreeTextTypingAgent
except ImportError:
        raise ImportError("Could not import FreeTextTypingAgent. Ensure the ABM typing simulator module is available.")

def setup_agent_profiles() -> Dict[str, Dict]:
    """
    Create distinct profiles for legitimate users and attackers.
    
    Returns:
        Dictionary of agent profiles with typing parameters
    """
    # Set random seed for reproducibility while still allowing parameterization
    random.seed(88)
    np.random.seed(88)
    
    # Create base profiles for legitimate users
    '''
    reference values for fatigue:
        (0.0001, 0.0005),  # Very slow fatigue
        (0.0005, 0.001),  # Slow fatigue
        (0.0015, 0.003),  # Quick fatigue
        (0.003, 0.005),  # Significant fatigue
        (0.005, 0.01),  # High fatigue
    '''
    profiles = {
        # Legitimate users with consistent typing patterns
        "user1": {
            "wpm_range": (50.0, 55.0),  # Moderate typing speed
            "error_rate_range": (0.03, 0.04),  # Low-moderate error rate
            "fatigue_factor_range": (0.0001, 0.0003),  # very slow fatigue
            "finger_agility_range": (0.9, 1.0),
            "dominant_hand": "left",
            "legitimate": True
        },
        "user2": {
            "wpm_range": (65.0, 70.0),  # Fast typist
            "error_rate_range": (0.01, 0.03),  # Very few errors
            "fatigue_factor_range": (0.0015, 0.003),  # Quick fatigue
            "finger_agility_range": (1.0, 1.1),
            "dominant_hand": "right",
            "legitimate": True
        },
        "user3": {
            "wpm_range": (40.0, 45.0),  # Slower, more deliberate typist
            "error_rate_range": (0.02, 0.03),  # Few errors
            "fatigue_factor_range": (0.0001, 0.0003),  # very slow fatigue
            "finger_agility_range": (0.8, 0.9),
            "dominant_hand": "right",
            "legitimate": True
        },
        
        # Attackers with divergent typing patterns
        "user4": {
            "wpm_range": (80.0, 85.0),  # Unusually fast
            "error_rate_range": (0.08, 0.1),  # Many errors
            "fatigue_factor_range": (0.0001, 0.0003),  # very slow fatigue
            "finger_agility_range": (1.2, 1.3),
            "dominant_hand": "right",
            "legitimate": False
        },
        "user5": {
            "wpm_range": (30.0, 35.0),  # Very slow
            "error_rate_range": (0.01, 0.02),  # Very Few errors
            "fatigue_factor_range":  (0.0001, 0.0003),  # very slow fatigue
            "finger_agility_range": (0.7, 0.8),
            "dominant_hand": "right",
            "legitimate": False
        }
    }
    
    return profiles

def generate_agent_data(
    agent_id: str, 
    profile: Dict, 
    base_path: str = "dataset",
    chars_per_file: int = 1000,
    add_session_variation: bool = True
) -> None:
    """
    Generate keystroke data files for a single agent across different keyboards and sessions.
    
    Parameters:
        agent_id (str): Identifier for the agent
        profile (Dict): Parameters defining the agent's typing characteristics
        base_path (str): Root directory for dataset storage
        chars_per_file (int): Number of characters to generate per file
        add_session_variation (bool): Whether to add realistic session-to-session variation
    """
    keyboards = ["laptop", "mechanical"]
    sessions = ["train", "test"]
    
    # Create output directory
    agent_dir = Path(base_path) / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating data for agent: {agent_id} ({'legitimate user' if profile['legitimate'] else 'attacker'})")
    
    # Generate a consistent base WPM for this agent
    base_wpm = np.random.uniform(*profile["wpm_range"])
    base_error_rate = np.random.uniform(*profile["error_rate_range"])
    base_fatigue = np.random.uniform(*profile["fatigue_factor_range"])
    finger_agility = np.random.uniform(*profile["finger_agility_range"])
    dominant_hand = profile["dominant_hand"]
    
    for keyboard in keyboards:
        print(f"  - Keyboard type: {keyboard}")
        
        # Adjust parameters slightly for different keyboards
        # Users typically type differently on different keyboard types
        keyboard_wpm_adjust = 1.1 if keyboard == "mechanical" else 1.0 # Study by Feit et al.(2016) found users typed 6–13% faster on mechanical keyboards compared to laptop keyboards.
        keyboard_error_adjust = 1.0 if keyboard == "mechanical" else 1.15 # Feit et al. (2016) observed 15% more corrections (backspaces) on laptop keyboards.
        
        for session_idx, session in enumerate(sessions):
            print(f"    - Session: {session}")
            
            # Add realistic session-to-session variation
            # People don't type exactly the same way each time
            if add_session_variation and session_idx > 0:
                session_wpm_factor = np.random.uniform(0.97, 1.03) # +/- 3%
                session_error_factor = np.random.uniform(0.97, 1.03) # +/- 3%
            else:
                session_wpm_factor = 1.0
                session_error_factor = 1.0
            
            # Calculate final parameters for this session
            wpm = base_wpm * keyboard_wpm_adjust * session_wpm_factor
            error_rate = base_error_rate * keyboard_error_adjust * session_error_factor
            fatigue_factor = base_fatigue
            
            # Create the agent with the specified parameters
            agent = FreeTextTypingAgent(
                agent_id=agent_id,
                wpm=wpm,
                error_rate=error_rate,
                keyboard_type=keyboard,
                fatigue_factor=fatigue_factor,
                finger_agility=finger_agility,
                dominant_hand=dominant_hand
            )
            
            # Generate keystrokes
            keystrokes = agent.generate_keystrokes(total_chars=chars_per_file)
            
            # Write to CSV file
            filename = agent_dir / f"{agent_id}_{keyboard}_{session}.csv"
            
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "timestamp_ms", 
                    "key", 
                    "action", 
                    "keyboard_type", 
                    "agent_id", 
                    "session_type",
                    "wpm",
                    "error_rate",
                    "fatigue_factor",
                    "finger_agility",
                    "dominant_hand"
                ])
                
                for event in keystrokes:
                    writer.writerow([
                        f"{event['time_ms']:.2f}",  # Format timestamp to 2 decimal places
                        event["key"],
                        event["action"],
                        keyboard,
                        agent_id,
                        session,
                        f"{wpm:.2f}",
                        f"{error_rate:.4f}",
                        f"{fatigue_factor:.6f}",
                        f"{finger_agility:.2f}",
                        f"{dominant_hand}"
                    ])
            
            print(f"      Created file: {filename}")

def create_dataset_info(base_path: str, profiles: Dict) -> None:
    """
    Create a metadata file describing the dataset contents.
    
    Parameters:
        base_path (str): Root directory for dataset storage
        profiles (Dict): Agent profiles used to generate the data
    """
    info_path = Path(base_path) / "dataset_info.txt"
    
    with open(info_path, "w") as f:
        f.write("ABM Typing Dataset\n")
        f.write("=================\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Dataset Structure:\n")
        f.write("- 5 Agents (3 legitimate users, 2 attackers)\n")
        f.write("- 4 files per agent (2 keyboard types × 2 sessions)\n")
        f.write("- CSV format with keystroke timing data\n\n")
        
        f.write("Agent Profiles:\n")
        for agent_id, profile in profiles.items():
            f.write(f"\n{agent_id} ({'Legitimate User' if profile['legitimate'] else 'Attacker'}):\n")
            f.write(f"  - WPM Range: {profile['wpm_range'][0]:.1f}-{profile['wpm_range'][1]:.1f}\n")
            f.write(f"  - Error Rate Range: {profile['error_rate_range'][0]:.3f}-{profile['error_rate_range'][1]:.3f}\n")
            f.write(f"  - Fatigue Factor Range: {profile['fatigue_factor_range'][0]:.6f}-{profile['fatigue_factor_range'][1]:.6f}\n")
    
    print(f"Created dataset info file: {info_path}")

def main(base_path: str = "dataset", chars_per_file: int = 1000):
    """
    Generate the complete dataset for all agents.
    
    Parameters:
        base_path (str): Root directory for dataset storage
        chars_per_file (int): Number of characters to generate per file
    """
    print(f"ABM Dataset Generator - Creating dataset in '{base_path}'")
    print(f"Generating {chars_per_file} characters per file\n")
    
    # Setup distinct profiles for different agent types
    profiles = setup_agent_profiles()
    
    # Create dataset directory
    dataset_dir = Path(base_path)
    dataset_dir.mkdir(exist_ok=True)
    
    # Generate data for all agents
    start_time = time.time()
    for agent_id, profile in profiles.items():
        generate_agent_data(
            agent_id=agent_id,
            profile=profile,
            base_path=base_path,
            chars_per_file=chars_per_file
        )
    
    # Create dataset info file
    create_dataset_info(base_path, profiles)
    
    elapsed_time = time.time() - start_time
    print(f"\nDataset generation complete in {elapsed_time:.2f} seconds.")
    print(f"Total files created: {len(profiles) * 2 * 2} (5 agents × 2 keyboards × 2 sessions)")
    print(f"Dataset location: {os.path.abspath(base_path)}")

if __name__ == "__main__":
    main()