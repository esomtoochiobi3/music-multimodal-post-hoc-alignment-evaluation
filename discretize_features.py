#!/usr/bin/env python3
"""
Feature Discretization for Caption Construction
Converts continuous features into categorical descriptors
"""

import numpy as np

# ============================================================================
# TEMPO DISCRETIZATION
# ============================================================================

def discretize_tempo(tempo_bpm):
    """
    Convert BPM to tempo descriptor
    
    Categories based on music theory:
    - Very Slow: < 60 BPM (Largo)
    - Slow: 60-90 BPM (Adagio, Andante)
    - Medium: 90-120 BPM (Moderato)
    - Fast: 120-168 BPM (Allegro)
    - Very Fast: > 168 BPM (Presto)
    """
    if tempo_bpm < 60:
        return "very slow"
    elif tempo_bpm < 90:
        return "slow"
    elif tempo_bpm < 120:
        return "medium-paced"
    elif tempo_bpm < 168:
        return "fast-paced"
    else:
        return "very fast"

# ============================================================================
# KEY FORMATTING
# ============================================================================

def format_key(key_str):
    """
    Ensure consistent key formatting
    
    Examples:
    - "Dm" → "D minor"
    - "C#" → "C# major"
    - "F" → "F major"
    """
    key_str = key_str.strip()
    
    # Check if minor (lowercase 'm' at end)
    if key_str.endswith('m'):
        root = key_str[:-1]
        return f"{root} minor"
    else:
        return f"{key_str} major"

# ============================================================================
# EMOTION DISCRETIZATION
# ============================================================================

def discretize_emotion(value, thresholds=None):
    """
    Convert continuous emotion score to categorical level
    
    Default thresholds (based on std dev ~0.5-0.8):
    - Low: < -0.3
    - Moderate: -0.3 to 0.3
    - High: > 0.3
    
    These are tuned for the Luo emotion model which outputs
    values roughly in [-2, 2] range with mean near 0
    """
    if thresholds is None:
        thresholds = {'low': -0.3, 'high': 0.3}
    
    if value < thresholds['low']:
        return "low"
    elif value > thresholds['high']:
        return "high"
    else:
        return "moderate"

# ============================================================================
# VALENCE-BASED MOOD MAPPING
# ============================================================================

def get_mood_descriptor(valence, energy, happy, sad):
    """
    Derive overall mood from multiple emotional dimensions
    
    Uses valence (positive/negative affect) as primary indicator,
    with energy, happy, and sad as modifiers
    
    Returns: (primary_mood, intensity)
    """
    
    # Primary mood from valence
    if valence > 0.3:
        # Positive valence
        if energy > 0.3:
            if happy > 0:
                primary = "uplifting and energetic"
            else:
                primary = "bright and vibrant"
        else:
            if happy > 0:
                primary = "cheerful and gentle"
            else:
                primary = "calm and pleasant"
    
    elif valence < -0.3:
        # Negative valence
        if energy > 0.3:
            if sad > 0:
                primary = "melancholic and intense"
            else:
                primary = "dark and aggressive"
        else:
            if sad > 0:
                primary = "somber and reflective"
            else:
                primary = "subdued and introspective"
    
    else:
        # Neutral valence
        if energy > 0.3:
            primary = "dynamic and neutral"
        else:
            primary = "ambient and atmospheric"
    
    return primary

# ============================================================================
# COMPLETE FEATURE DISCRETIZATION
# ============================================================================

def discretize_all_features(track_features):
    """
    Discretize all features for a single track
    
    Input: dict with keys:
        - tempo (float)
        - key (str)
        - valence, energy, tension, anger, fear, happy, sad, tender (floats)
    
    Output: dict with discretized features
    """
    discretized = {}
    
    # Structural features
    discretized['tempo_desc'] = discretize_tempo(track_features['tempo'])
    discretized['key_formatted'] = format_key(track_features['key'])
    
    # Emotion levels
    emotions = ['valence', 'energy', 'tension', 'anger', 'fear', 'happy', 'sad', 'tender']
    for emotion in emotions:
        if emotion in track_features:
            discretized[f'{emotion}_level'] = discretize_emotion(track_features[emotion])
    
    # Derived mood
    discretized['mood'] = get_mood_descriptor(
        track_features['valence'],
        track_features['energy'],
        track_features['happy'],
        track_features['sad']
    )
    
    # Keep original values for reference
    discretized['original'] = track_features.copy()
    
    return discretized

# ============================================================================
# TESTING / EXAMPLES
# ============================================================================

if __name__ == '__main__':
    # Test cases
    
    print("=" * 70)
    print("FEATURE DISCRETIZATION EXAMPLES")
    print("=" * 70)
    
    # Example 1: Happy, upbeat track
    track1 = {
        'tempo': 128.5,
        'key': 'C',
        'valence': 0.54,
        'energy': 0.37,
        'tension': -0.26,
        'anger': -0.95,
        'fear': -0.81,
        'happy': 0.72,
        'sad': -0.88,
        'tender': -0.06
    }
    
    print("\nExample 1: Happy, upbeat track")
    print("-" * 70)
    result1 = discretize_all_features(track1)
    print(f"Tempo: {track1['tempo']:.1f} BPM → {result1['tempo_desc']}")
    print(f"Key: {track1['key']} → {result1['key_formatted']}")
    print(f"Valence: {track1['valence']:.2f} → {result1['valence_level']}")
    print(f"Energy: {track1['energy']:.2f} → {result1['energy_level']}")
    print(f"Happy: {track1['happy']:.2f} → {result1['happy_level']}")
    print(f"Mood: {result1['mood']}")
    
    # Example 2: Dark, intense track
    track2 = {
        'tempo': 85.2,
        'key': 'Dm',
        'valence': -0.59,
        'energy': 0.48,
        'tension': 0.83,
        'anger': 0.75,
        'fear': 1.03,
        'happy': -1.38,
        'sad': -1.20,
        'tender': -1.21
    }
    
    print("\nExample 2: Dark, intense track")
    print("-" * 70)
    result2 = discretize_all_features(track2)
    print(f"Tempo: {track2['tempo']:.1f} BPM → {result2['tempo_desc']}")
    print(f"Key: {track2['key']} → {result2['key_formatted']}")
    print(f"Valence: {track2['valence']:.2f} → {result2['valence_level']}")
    print(f"Energy: {track2['energy']:.2f} → {result2['energy_level']}")
    print(f"Tension: {track2['tension']:.2f} → {result2['tension_level']}")
    print(f"Mood: {result2['mood']}")
    
    # Example 3: Calm, ambient track
    track3 = {
        'tempo': 72.0,
        'key': 'F#m',
        'valence': -0.23,
        'energy': -0.34,
        'tension': 0.16,
        'anger': -1.15,
        'fear': -0.33,
        'happy': -0.99,
        'sad': 0.83,
        'tender': 0.16
    }
    
    print("\nExample 3: Calm, melancholic track")
    print("-" * 70)
    result3 = discretize_all_features(track3)
    print(f"Tempo: {track3['tempo']:.1f} BPM → {result3['tempo_desc']}")
    print(f"Key: {track3['key']} → {result3['key_formatted']}")
    print(f"Valence: {track3['valence']:.2f} → {result3['valence_level']}")
    print(f"Energy: {track3['energy']:.2f} → {result3['energy_level']}")
    print(f"Sad: {track3['sad']:.2f} → {result3['sad_level']}")
    print(f"Mood: {result3['mood']}")
    
    print("\n" + "=" * 70)
    print("Test complete!")