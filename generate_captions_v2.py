#!/usr/bin/env python3
"""
Caption Generation with Genre Tags
Integrates cleaned genre tags from metadata into theory-enriched captions
"""

import random
from discretize_features import discretize_all_features
from clean_tags import clean_tag_string, format_genres_for_caption

# ============================================================================
# CAPTION TEMPLATES WITH GENRE SUPPORT
# ============================================================================

# Full captions (structural + affective + genre)
FULL_TEMPLATES_WITH_GENRE = [
    "A {tempo} track in {key} with {genre_phrase}. Musically, it features {energy_desc} and {mood}, creating a {overall_feel} atmosphere.",
    
    "This {genre_phrase} piece is in {key} with a {tempo} pace. It exhibits {energy_desc} and {tension_desc}, evoking a {mood} feeling.",
    
    "Set in {key}, this {tempo} {genre_phrase} composition conveys {mood} through its {energy_desc} character.",
    
    "A {tempo} {genre_phrase} song in {key}, characterized by {energy_desc}. The {mood} quality creates a {overall_feel} experience.",
    
    "In {key}, this {tempo} {genre_phrase} track blends {energy_desc} with {mood}, resulting in a {overall_feel} sonic landscape.",
]

# Structural-only templates (key + tempo + genre)
STRUCTURAL_TEMPLATES_WITH_GENRE = [
    "A {tempo} track in {key}. Genre: {genre}.",
    
    "This {genre_phrase} piece is in {key} with a {tempo} pace.",
    
    "Set in {key}, this {tempo} {genre_phrase} composition.",
    
    "A {tempo} {genre_phrase} song in {key}.",
]

# Affective-only templates (emotions + genre)
AFFECTIVE_TEMPLATES_WITH_GENRE = [
    "A {genre_phrase} track featuring {energy_desc} with {mood}, creating a {overall_feel} atmosphere.",
    
    "This {genre_phrase} piece exhibits {energy_desc} and {tension_desc}, evoking a {mood} feeling.",
    
    "A {genre_phrase} composition that conveys {mood} through its {energy_desc} character.",
    
    "This {genre_phrase} song is characterized by {energy_desc}. The {mood} quality creates a {overall_feel} experience.",
]

# Fallback templates (no genre available)
FULL_TEMPLATES_NO_GENRE = [
    "A {tempo} track in {key}. Musically, it features {energy_desc} and {mood}, creating a {overall_feel} atmosphere.",
    
    "This piece is in {key} with a {tempo} pace. It exhibits {energy_desc} and {tension_desc}, evoking a {mood} feeling.",
    
    "Set in {key}, this {tempo} composition conveys {mood} through its {energy_desc} character.",
]

STRUCTURAL_TEMPLATES_NO_GENRE = [
    "A {tempo} track in {key}.",
    "This piece is in {key} with a {tempo} pace.",
]

AFFECTIVE_TEMPLATES_NO_GENRE = [
    "A track featuring {energy_desc} with {mood}, creating a {overall_feel} atmosphere.",
    "This piece exhibits {energy_desc} and {tension_desc}, evoking a {mood} feeling.",
]

# ============================================================================
# DESCRIPTOR FUNCTIONS (from original generate_captions.py)
# ============================================================================

def get_energy_descriptor(energy_level, energy_val):
    """Generate energy description with variation"""
    descriptors = {
        'low': [
            "low energy and gentle dynamics",
            "subdued energy",
            "calm intensity",
            "restrained power"
        ],
        'moderate': [
            "moderate energy",
            "balanced dynamics",
            "steady intensity",
            "measured power"
        ],
        'high': [
            "high energy and powerful dynamics",
            "intense energy",
            "strong intensity",
            "vigorous power"
        ]
    }
    return random.choice(descriptors[energy_level])

def get_tension_descriptor(tension_level):
    """Generate tension description"""
    descriptors = {
        'low': [
            "minimal tension",
            "relaxed harmonic content",
            "consonant structure",
            "stable tonality"
        ],
        'moderate': [
            "moderate tension",
            "balanced harmonic content",
            "mixed consonance",
            "dynamic tonality"
        ],
        'high': [
            "high tension",
            "dissonant harmonic content",
            "unstable structure",
            "complex tonality"
        ]
    }
    return random.choice(descriptors[tension_level])

def get_valence_descriptor(valence_level):
    """Generate valence description"""
    descriptors = {
        'low': [
            "negative emotional quality",
            "dark timbral character",
            "somber tonal palette",
            "melancholic essence"
        ],
        'moderate': [
            "neutral emotional quality",
            "balanced timbral character",
            "ambiguous tonal palette",
            "contemplative essence"
        ],
        'high': [
            "positive emotional quality",
            "bright timbral character",
            "uplifting tonal palette",
            "optimistic essence"
        ]
    }
    return random.choice(descriptors[valence_level])

def get_overall_feel(mood):
    """Map mood to overall feel descriptor"""
    mood_to_feel = {
        "uplifting and energetic": ["dynamic", "exhilarating", "vibrant", "lively"],
        "bright and vibrant": ["radiant", "spirited", "animated", "vivid"],
        "cheerful and gentle": ["pleasant", "warm", "welcoming", "comforting"],
        "calm and pleasant": ["serene", "peaceful", "tranquil", "soothing"],
        "melancholic and intense": ["dramatic", "poignant", "evocative", "moving"],
        "dark and aggressive": ["intense", "powerful", "forceful", "commanding"],
        "somber and reflective": ["introspective", "contemplative", "meditative", "pensive"],
        "subdued and introspective": ["thoughtful", "gentle", "understated", "nuanced"],
        "dynamic and neutral": ["balanced", "versatile", "adaptive", "multifaceted"],
        "ambient and atmospheric": ["ethereal", "spacious", "immersive", "textural"]
    }
    
    return random.choice(mood_to_feel.get(mood, ["distinctive", "unique", "characteristic"]))

# ============================================================================
# CAPTION GENERATION WITH GENRE INTEGRATION
# ============================================================================

def generate_caption(track_features, metadata_tags=None, caption_type='full'):
    """
    Generate a natural language caption for a track with genre tags
    
    Args:
        track_features: dict with continuous features (from CSV)
        metadata_tags: str with raw metadata tags (e.g., "rock, alternative, metal")
        caption_type: 'full', 'structural', or 'affective'
    
    Returns:
        caption string
    """
    # Discretize features
    disc = discretize_all_features(track_features)
    
    # Clean genre tags
    genre_tags = clean_tag_string(metadata_tags) if metadata_tags else []
    genre_formatted = format_genres_for_caption(genre_tags)
    has_genre = genre_formatted is not None
    
    # Select template based on type and genre availability
    if caption_type == 'structural':
        if has_genre:
            template = random.choice(STRUCTURAL_TEMPLATES_WITH_GENRE)
        else:
            template = random.choice(STRUCTURAL_TEMPLATES_NO_GENRE)
    
    elif caption_type == 'affective':
        if has_genre:
            template = random.choice(AFFECTIVE_TEMPLATES_WITH_GENRE)
        else:
            template = random.choice(AFFECTIVE_TEMPLATES_NO_GENRE)
    
    else:  # 'full'
        if has_genre:
            template = random.choice(FULL_TEMPLATES_WITH_GENRE)
        else:
            template = random.choice(FULL_TEMPLATES_NO_GENRE)
    
    # Build descriptor dictionary
    descriptors = {
        'tempo': disc['tempo_desc'],
        'key': disc['key_formatted'],
        'mood': disc['mood'],
        'energy_desc': get_energy_descriptor(disc['energy_level'], disc['original']['energy']),
        'tension_desc': get_tension_descriptor(disc['tension_level']),
        'valence_desc': get_valence_descriptor(disc['valence_level']),
        'overall_feel': get_overall_feel(disc['mood']),
        'genre': genre_formatted if has_genre else "",
        'genre_phrase': genre_formatted if has_genre else "musical"
    }
    
    # Fill template
    try:
        caption = template.format(**descriptors)
    except KeyError as e:
        # Fallback if template requires missing key
        if has_genre:
            caption = f"A {descriptors['tempo']} {genre_formatted} track in {descriptors['key']}."
        else:
            caption = f"A {descriptors['tempo']} track in {descriptors['key']}."
    
    return caption

def generate_redacted_caption(original_prompt, metadata_tags=None):
    """
    Create redacted baseline caption by removing structural information
    from original Suno user prompts
    
    Removes: key names, BPM numbers, "major", "minor", tempo terms
    Keeps: genre (from tags), qualitative descriptors, style information
    """
    import re
    
    # Start with original prompt
    caption = original_prompt if original_prompt else ""
    
    # Remove key signatures (C, D, E, F, G, A, B with sharps/flats and major/minor)
    caption = re.sub(r'\b[A-G][#b]?\s*(major|minor|maj|min)\b', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\b(in\s+)?[A-G][#b]?\b', '', caption)  # Remove standalone keys
    
    # Remove BPM references
    caption = re.sub(r'\d+\s*(bpm|beats per minute)', '', caption, flags=re.IGNORECASE)
    
    # Remove tempo descriptors that are structural
    tempo_terms = ['adagio', 'andante', 'moderato', 'allegro', 'presto', 'largo', 'vivace']
    for term in tempo_terms:
        caption = re.sub(rf'\b{term}\b', '', caption, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    # Add genre tags if available
    if metadata_tags:
        genre_tags = clean_tag_string(metadata_tags)
        genre_formatted = format_genres_for_caption(genre_tags)
        if genre_formatted:
            if caption:
                caption = f"{caption}. Genre: {genre_formatted}"
            else:
                caption = f"Genre: {genre_formatted}"
    
    return caption if caption else "A musical composition."

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def generate_captions_batch(tracks_df, caption_type='full'):
    """
    Generate captions for a batch of tracks from a DataFrame
    
    Args:
        tracks_df: pandas DataFrame with feature columns and optional 'tags' column
        caption_type: 'full', 'structural', or 'affective'
    
    Returns:
        list of captions (same order as input)
    """
    captions = []
    
    for idx, row in tracks_df.iterrows():
        track_features = row.to_dict()
        metadata_tags = row.get('tags', None)
        caption = generate_caption(track_features, metadata_tags, caption_type)
        captions.append(caption)
    
    return captions

# ============================================================================
# TESTING / EXAMPLES
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("CAPTION GENERATION WITH GENRE TAGS")
    print("=" * 80)
    
    # Example track features
    track = {
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
    
    # Test with different genre tags
    test_cases = [
        ("electronic, techno, house", "Clean electronic genres"),
        ("folk song", "Simple folk"),
        ("alternative metal, dark and heavy riffs", "Genre + descriptors"),
        ("", "No genres"),
        (None, "Missing tags")
    ]
    
    for tags, description in test_cases:
        print(f"\n{description}")
        print(f"Tags: {tags}")
        print("-" * 80)
        
        print("\nFull caption:")
        print(f"  {generate_caption(track, tags, 'full')}")
        
        print("\nStructural only:")
        print(f"  {generate_caption(track, tags, 'structural')}")
        
        print("\nAffective only:")
        print(f"  {generate_caption(track, tags, 'affective')}")
        
        print()
    
    # Test redacted baseline
    print("\n" + "=" * 80)
    print("REDACTED BASELINE EXAMPLES")
    print("=" * 80)
    
    original = "A fast electronic track in D minor at 140 BPM with aggressive synths"
    tags = "electronic, techno, edm"
    redacted = generate_redacted_caption(original, tags)
    print(f"Original: {original}")
    print(f"Tags: {tags}")
    print(f"Redacted: {redacted}")
    
    print("\n" + "=" * 80)
    print("Test complete!")