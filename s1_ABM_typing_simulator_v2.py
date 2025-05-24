# -*- coding: utf-8 -*-
'''
eStyler v.2.0.0

Agent Based Modeling (ABM) of User typing
(enhanced version with key-specific flight times)

by Roberto Dillon, (c) 2025


This simulator generates realistic typing patterns for different user types,
including variations in typing speed, error rates, and keyboard characteristics.

Key Features:
- Free-Text Generation: Creates English-like text using character/word statistics
- Multi-Backspace Errors: Simulates 1-6 backspaces with realistic timing
- Contextual Pauses: Includes natural pauses between words
- Realistic Fatigue: Non-linear slowdown as typing continues
- Key-Specific Flight Times: Models realistic timing between specific key pairs
- Individual User Profiles: Includes personal typing patterns that vary by user

Example usage:

# Initialize agents
professor_agent = FreeTextTypingAgent(
    agent_id=1, 
    wpm=40.0, 
    error_rate=0.03,
    keyboard_type="mechanical",
    finger_agility=0.8  # Professor has slower finger movements
)

student_agent = FreeTextTypingAgent(
    agent_id=2,
    wpm=65.0,
    error_rate=0.1,
    keyboard_type="laptop",
    finger_agility=1.2  # Student has faster finger movements
)

# Generate 500 keystrokes
professor_keystrokes = professor_agent.generate_keystrokes(total_chars=500)
student_keystrokes = student_agent.generate_keystrokes(total_chars=500)

# Inspect output
print(f"Professor typed: {len(professor_keystrokes)//2} chars")
print(f"Student typed: {len(student_keystrokes)//2} chars")

------

v 2.2

The typing simulator was enhanced to include keyboard-specific flight times between 
key pairs (based on the spatial relationship between keys on a QWERTY keyboard), 
this version adds a more realistic dimension to the agent's typing behavior.

The key additions are:

A representation of key distances on a QWERTY keyboard
A mechanism to use these distances to calculate flight times between specific key pairs
Integration of this system with the existing WPM, fatigue, and randomization factors
 
The key enhancements made to the typing simulator include:
1. QWERTY Keyboard Physical Layout
    Added a detailed representation of the QWERTY keyboard layout with precise coordinates for each key. 
    This allows the simulation to calculate the actual physical distance between any two keys.
2. Key-Specific Flight Times
    The agent now calculates flight times between keys based on:

    The physical distance between keys (closer keys are faster to type)
    The user's personal typing patterns
    Special handling for repeated keys (typing the same key twice is much faster)

3. Personal Typing Pattern Variations
    Each agent now has unique typing characteristics that reflect individual typing behaviors:

    Hand dominance affects typing speed on different sides of the keyboard
    Users type common letter combinations (digraphs like "th", "er", "on") faster
    Each agent has randomly generated strengths and weaknesses on specific key pairs

4. Extended User Configuration Options
    Added several parameters that let you create more distinctive user profiles:

    finger_agility: How quickly a user can move between keys
    dominant_hand: "right" or "left" to model hand dominance effects
    keyboard_layout: Currently supports "qwerty" (extendable to other layouts)

5. Complete Integration with Existing Features
The new key-specific flight time system works seamlessly with the existing:

WPM-based typing speed
Fatigue modeling
Error and backspace simulation
Word pauses and text generation

This enhanced simulator should now produce typing patterns that are much more characteristic of individual users, 
making it much more suitable for your machine learning classification project.


'''

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class FreeTextTypingAgent:
    """
    Agent that simulates realistic human typing behavior.
    
    This class models a person typing on a keyboard with individual 
    characteristics including typing speed, error frequency, fatigue patterns,
    and key-specific flight times.
    """
    def __init__(
        self,
        agent_id: int,
        wpm: float = 45.0,
        error_rate: float = 0.05,
        keyboard_type: str = "laptop",
        fatigue_factor: float = 0.001,
        finger_agility: float = 1.0,
        dominant_hand: str = "right",
        keyboard_layout: str = "qwerty"
    ):
        """
        Initialize a typing agent with specific characteristics.
        
        Parameters:
            agent_id (int): Unique identifier for the agent
            wpm (float): Words per minute typing speed (base/starting speed)
            error_rate (float): Probability of making typing errors (0.0 to 1.0)
            keyboard_type (str): "mechanical" or "laptop" - affects key timings
            fatigue_factor (float): Rate at which typing speed decreases over time
            finger_agility (float): Multiplier for finger movement speed (>1 faster, <1 slower)
            dominant_hand (str): "right" or "left" - affects typing patterns
            keyboard_layout (str): Currently only "qwerty" is supported
        """
        self.agent_id = agent_id
        self.base_wpm = wpm
        self.error_rate = error_rate
        self.keyboard_type = keyboard_type
        self.fatigue_factor = fatigue_factor
        self.finger_agility = finger_agility
        self.dominant_hand = dominant_hand
        self.keyboard_layout = keyboard_layout
        self.current_wpm = wpm  # Current WPM will decrease with fatigue
        self.fatigue = 0.0      # Increases as the agent types
        
        # English language statistics for text generation
        self.char_distribution = self._load_char_distribution()
        self.avg_word_length = 4.7  # Average English word length
        
        # Set keyboard-specific timing profiles
        self._set_keyboard_profile()
        
        # Initialize key distance matrix for the keyboard layout
        self.key_distances = self._initialize_key_distances()
        
        # Initialize personal typing pattern variations
        self.personal_variation = self._initialize_personal_pattern()
    
    def _set_keyboard_profile(self) -> None:
        """
        Set key press/release timing profiles based on keyboard type.
        
        Different keyboards have different physical characteristics affecting typing:
        - Mechanical: Longer key travel, distinct tactile feedback
        - Laptop/membrane: Shorter key travel, less distinct feedback
        """
        if self.keyboard_type == "mechanical":
            # Mechanical keyboards typically have longer dwell times (key press duration)
            # but shorter flight times (time between keys) due to tactile feedback
            self.base_dwell = 60.0  # milliseconds
            self.base_flight = 100.0  # milliseconds
        else:  # laptop/membrane
            # Laptop keyboards have shorter key travel but often require more
            # time between keypresses due to less distinct feedback
            self.base_dwell = 50.0  # milliseconds
            self.base_flight = 120.0  # milliseconds
    
    def _load_char_distribution(self) -> Dict[str, float]:
        """
        Load approximate frequency distribution of characters in English text.
        
        Returns:
            Dict mapping characters to their frequency probabilities (0.0 to 1.0)
        """
        raw_frequencies = {
            'a': 0.0817, 'b': 0.0149, 'c': 0.0278, 'd': 0.0425, 'e': 0.1270,
            'f': 0.0223, 'g': 0.0202, 'h': 0.0609, 'i': 0.0697, 'j': 0.0015,
            'k': 0.0077, 'l': 0.0403, 'm': 0.0241, 'n': 0.0675, 'o': 0.0751,
            'p': 0.0193, 'q': 0.0010, 'r': 0.0599, 's': 0.0633, 't': 0.0906,
            'u': 0.0276, 'v': 0.0098, 'w': 0.0236, 'x': 0.0015, 'y': 0.0197,
            'z': 0.0007, ' ': 0.1300  # Space
        }
        total = sum(raw_frequencies.values())
        return {k: v / total for k, v in raw_frequencies.items()}
    
    def _initialize_key_distances(self) -> Dict[Tuple[str, str], float]:
        """
        Initialize a matrix of distances between key pairs on the keyboard.
        
        Returns:
            Dictionary mapping key pairs (char1, char2) to their distance values
        """
        if self.keyboard_layout != "qwerty":
            raise ValueError(f"Keyboard layout '{self.keyboard_layout}' not supported")
        
        # Define QWERTY keyboard layout with coordinates (row, column)
        # Row 0 is the top row (numbers), row 3 is the bottom row
        qwerty_positions = {
            '`': (0, 0), '1': (0, 1), '2': (0, 2), '3': (0, 3), '4': (0, 4), '5': (0, 5),
            '6': (0, 6), '7': (0, 7), '8': (0, 8), '9': (0, 9), '0': (0, 10), '-': (0, 11),
            '=': (0, 12),
            'q': (1, 1.5), 'w': (1, 2.5), 'e': (1, 3.5), 'r': (1, 4.5), 't': (1, 5.5),
            'y': (1, 6.5), 'u': (1, 7.5), 'i': (1, 8.5), 'o': (1, 9.5), 'p': (1, 10.5),
            '[': (1, 11.5), ']': (1, 12.5), '\\': (1, 13.5),
            'a': (2, 1.75), 's': (2, 2.75), 'd': (2, 3.75), 'f': (2, 4.75), 'g': (2, 5.75),
            'h': (2, 6.75), 'j': (2, 7.75), 'k': (2, 8.75), 'l': (2, 9.75), ';': (2, 10.75),
            "'": (2, 11.75),
            'z': (3, 2.25), 'x': (3, 3.25), 'c': (3, 4.25), 'v': (3, 5.25), 'b': (3, 6.25),
            'n': (3, 7.25), 'm': (3, 8.25), ',': (3, 9.25), '.': (3, 10.25), '/': (3, 11.25),
            ' ': (4, 7)  # Space bar is centered at the bottom
        }
        
        # Calculate Euclidean distances between all key pairs
        distances = {}
        all_keys = qwerty_positions.keys()
        
        for k1 in all_keys:
            for k2 in all_keys:
                if k1 == k2:
                    # Same key: very short distance (for double-letter typing)
                    distances[(k1, k2)] = 0.2
                else:
                    # Calculate Euclidean distance
                    pos1 = qwerty_positions[k1]
                    pos2 = qwerty_positions[k2]
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    distances[(k1, k2)] = distance
        
        return distances
    
    def _initialize_personal_pattern(self) -> Dict[Tuple[str, str], float]:
        """
        Create personalized variation in key-pair typing speed.
        
        Each user has slightly different strengths and weaknesses in finger movement patterns.
        This function creates a multiplier for each key pair, representing how this particular
        user deviates from the average.
        
        Returns:
            Dictionary mapping key pairs to personal speed multipliers
        """
        # Start with an empty pattern
        pattern = {}
        
        # Get all possible key pairs from the distance matrix
        key_pairs = self.key_distances.keys()
        
        # Apply personal variations to each key pair
        for k1, k2 in key_pairs:
            # Base variation is centered around 1.0 (no change)
            # We use a normal distribution with 15% standard deviation
            variation = max(0.7, min(1.3, np.random.normal(1.0, 0.15)))
            
            # Apply dominant hand advantage/disadvantage
            if self.dominant_hand == "right":
                right_hand_keys = "yuiophjklnm"
                left_hand_keys = "qwertasdfgzxcvb"
                if k1 in right_hand_keys and k2 in right_hand_keys:
                    # Right-handed users are faster with right-hand keys
                    variation *= 0.9  # 10% faster
                elif k1 in left_hand_keys and k2 in left_hand_keys:
                    # And slightly slower with left-hand keys
                    variation *= 1.05  # 5% slower
            else:  # left-dominant
                # Opposite effect for left-handed users
                right_hand_keys = "yuiophjklnm"
                left_hand_keys = "qwertasdfgzxcvb"
                if k1 in left_hand_keys and k2 in left_hand_keys:
                    variation *= 0.9  # 10% faster 
                elif k1 in right_hand_keys and k2 in right_hand_keys:
                    variation *= 1.05  # 5% slower
            
            # Apply "common digraphs" advantage - users are faster at typing common letter pairs
            common_digraphs = [('t', 'h'), ('h', 'e'), ('i', 'n'), ('e', 'r'), ('a', 'n'),
                              ('r', 'e'), ('e', 's'), ('o', 'n'), ('s', 't'), ('n', 't'),
                              ('e', 'n'), ('a', 't'), ('e', 'd'), ('n', 'd'), ('t', 'o'),
                              ('l', 'l'), ('a', 'n'), ('s', 's'), ('e', 'e')]
            if (k1, k2) in common_digraphs or (k2, k1) in common_digraphs:
                variation *= 0.85  # 15% faster on common letter combinations
            
            pattern[(k1, k2)] = variation
            
        return pattern
    
    def _generate_english_char(self) -> str:
        """
        Sample a character based on English letter frequency distribution.
        
        Returns:
            A single character (letter or space)
        """
        chars = list(self.char_distribution.keys())
        probs = list(self.char_distribution.values())
        return np.random.choice(chars, p=probs)
    
    def _generate_word(self) -> str:
        """
        Generate a random word with length based on English word length distribution.
        
        The length is drawn from a normal distribution centered at the average
        English word length (4.7 characters).
        
        Returns:
            A randomly generated word
        """
        length = max(1, int(np.random.normal(self.avg_word_length, 1.5)))
        # Filter out spaces to ensure we're generating actual word characters
        word_chars = []
        for _ in range(length):
            char = self._generate_english_char()
            while char == ' ':  # Avoid spaces within words
                char = self._generate_english_char()
            word_chars.append(char)
        return ''.join(word_chars)
    
    def _simulate_backspaces(self, current_time: float) -> List[Dict]:
        """
        Simulate a series of backspace keypresses to correct typing errors.
        
        Parameters:
            current_time (float): Current timestamp in milliseconds
            
        Returns:
            List of keypress events for backspace actions
        """
        n_backspaces = random.randint(1, 6)  # Random number of backspaces (1-6)
        events = []
        
        # Backspaces typically happen quickly as the user realizes and corrects an error
        for _ in range(n_backspaces):
            # Faster dwell time for backspace corrections
            dwell = max(10.0, np.random.normal(40.0, 3.0))  
            events.extend([
                {"key": "[BACKSPACE]", "action": "press", "time_ms": current_time},
                {"key": "[BACKSPACE]", "action": "release", "time_ms": current_time + dwell}
            ])
            # Very short delays between consecutive backspaces
            current_time += dwell + max(5.0, np.random.normal(30.0, 5.0))
        return events
    
    def _get_key_pair_flight_time(self, prev_key: str, current_key: str) -> float:
        """
        Calculate the flight time between two specific keys.
        
        This considers:
        1. Physical distance between keys on the keyboard
        2. User's personal typing pattern
        3. Current fatigue level
        4. Random variation (noise)
        
        Parameters:
            prev_key (str): The previous key that was pressed
            current_key (str): The current key being pressed
            
        Returns:
            float: Flight time in milliseconds
        """
        # Handle case sensitivity by converting to lowercase
        prev_key = prev_key.lower()
        current_key = current_key.lower()
        
        # Default distance if key pair not found
        default_distance = 1.0
        
        # Get physical distance between keys (normalized to 0-2 range)
        # Use default if the key pair isn't in our distance map
        distance = self.key_distances.get((prev_key, current_key), default_distance)
        
        # Get personal variation for this key pair (defaults to 1.0 if not found)
        personal_factor = self.personal_variation.get((prev_key, current_key), 1.0)
        
        # Calculate base flight time:
        # - More distant keys take longer
        # - Personal pattern affects speed
        # - User's finger agility affects overall speed
        base_flight_time = self.base_flight * (0.5 + distance/2) * personal_factor / self.finger_agility
        
        # Apply current fatigue level (non-linear effect)
        fatigued_flight_time = base_flight_time * (1.0 + 0.4 * (self.fatigue ** 2))
        
        # Add random noise (5% standard deviation)
        noise_factor = max(0.9, min(1.1, np.random.normal(1.0, 0.05)))
        
        # Return final flight time with noise
        return fatigued_flight_time * noise_factor
    
    def _get_keystroke_timings(self, prev_key: Optional[str] = None, current_key: Optional[str] = None) -> Tuple[float, float]:
        """
        Generate realistic dwell and flight times for keypresses.
        
        Incorporates typing speed, key-specific patterns, and fatigue effects.
        
        Parameters:
            prev_key (str, optional): The previous key pressed
            current_key (str, optional): The current key being pressed
            
        Returns:
            Tuple of (dwell_time, flight_time) in milliseconds
        """
        # Adjust timing based on current WPM vs base WPM
        speed_factor = self.base_wpm / self.current_wpm
        
        # Dwell time (key press duration)
        dwell = max(10.0, np.random.normal(
            loc=self.base_dwell * speed_factor,
            scale=5.0  # Small variation in press time
        ))
        
        # Flight time calculation
        if prev_key is not None and current_key is not None:
            # Use key-specific flight time
            flight = self._get_key_pair_flight_time(prev_key, current_key)
        else:
            # Fallback to general flight time if keys not specified
            flight = max(5.0, np.random.normal(
                loc=self.base_flight * speed_factor,
                scale=15.0  # Larger variation in inter-key time
            ))
        
        return dwell, flight
    
    def _apply_fatigue(self) -> None:
        """
        Apply non-linear fatigue effect to reduce typing speed over time.
        
        Fatigue accumulates with each keystroke and has an increasing impact
        on typing speed (quadratic relationship for realism).
        """
        # Increase fatigue by a small amount
        self.fatigue = min(1.0, self.fatigue + self.fatigue_factor)
        
        # Apply non-linear (quadratic) fatigue effect to typing speed
        # At max fatigue (1.0), typing speed is reduced by 30%
        self.current_wpm = self.base_wpm * (1.0 - 0.3 * (self.fatigue ** 2))
    
    def generate_keystrokes(self, total_chars: int = 300) -> List[Dict]:
        """
        Simulate typing activity and generate a sequence of keystroke events.
        
        Parameters:
            total_chars (int): Target number of characters to generate
            
        Returns:
            List of keystroke events with timestamps
        """
        events = []
        current_time = 0.0
        generated_text = ""
        last_key = None  # Track the last key pressed
        
        # Safety counter to prevent infinite loops with high error rates
        safety_counter = 0
        max_iterations = total_chars * 10  # Reasonable upper bound
        
        while len(generated_text) < total_chars and safety_counter < max_iterations:
            safety_counter += 1
            
            # Generate a random word
            word = self._generate_word()
            
            for char in word:
                # Check if we've reached the target character count
                if len(generated_text) >= total_chars:
                    break
                    
                # Randomly inject typing errors based on error rate
                if random.random() < self.error_rate and generated_text:
                    # Add backspace events to simulate error correction
                    backspace_events = self._simulate_backspaces(current_time)
                    events.extend(backspace_events)
                    
                    # Update current time to after the backspaces
                    current_time = backspace_events[-1]["time_ms"] + 5.0  # Small delay after correction
                    
                    # Remove characters from generated text to match backspaces
                    # (limited by the actual text length to avoid issues)
                    chars_to_remove = min(len(generated_text), len(backspace_events) // 2)
                    generated_text = generated_text[:-chars_to_remove]
                    
                    # Update last key to be the key before the error (if any text left)
                    if generated_text:
                        last_key = generated_text[-1]
                    else:
                        last_key = None
                
                # Calculate key-specific timings
                dwell, flight = self._get_keystroke_timings(last_key, char)
                
                # Type the character (press and release events)
                events.extend([
                    {"key": char, "action": "press", "time_ms": current_time},
                    {"key": char, "action": "release", "time_ms": current_time + dwell}
                ])
                
                # Update timing and add character to generated text
                current_time += dwell + flight
                generated_text += char
                last_key = char  # Update the last key for next iteration
                
                # Apply fatigue effect after each character
                self._apply_fatigue()
            
            # Add space after word (if we haven't reached the character limit)
            if len(generated_text) < total_chars:
                # Calculate key-specific timing for space
                space_dwell, _ = self._get_keystroke_timings(last_key, " ")
                
                # Add spacebar events
                events.extend([
                    {"key": " ", "action": "press", "time_ms": current_time},
                    {"key": " ", "action": "release", "time_ms": current_time + space_dwell}
                ])
                
                # Add longer pause after words (exponential distribution for natural pauses)
                word_pause = np.random.exponential(200.0)  # ~200ms average pause between words
                current_time += space_dwell + word_pause
                
                # Add space to the generated text
                generated_text += " "
                last_key = " "  # Update last key
        
        # Ensure we have the exact number of characters
        # Each character has 2 events (press and release)
        target_events = total_chars * 2
        
        # If we generated too many events, trim the excess
        if len(events) > target_events:
            events = events[:target_events]
            
        # If we have too few events (rare with safety counter), duplicate some
        while len(events) < target_events:
            # Find a random key press/release pair to duplicate
            idx = random.randrange(0, len(events) - 1, 2)
            press_event = events[idx].copy()
            release_event = events[idx + 1].copy()
            
            # Add a small time offset to avoid exact duplicates
            time_offset = max(events[-1]["time_ms"] + 100, current_time + 100)
            press_event["time_ms"] = time_offset
            release_event["time_ms"] = time_offset + random.uniform(40, 80)
            
            events.extend([press_event, release_event])
        
        return events