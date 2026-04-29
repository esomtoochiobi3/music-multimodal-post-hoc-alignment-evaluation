#!/usr/bin/env python3
"""
Tag Cleaning Module for Suno Metadata
Extracts clean genre tags from noisy user-provided metadata
"""

import re
import string

# ============================================================================
# LANGUAGE DETECTION & FILTERING
# ============================================================================

def is_likely_english(text):
    """
    Simple heuristic to detect if text is likely English
    Checks for common English words and ASCII characters
    """
    if not text:
        return False
    
    # Check if mostly ASCII
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / len(text) < 0.8:
        return False
    
    # Common English music genre words
    english_indicators = [
        'the', 'and', 'with', 'for', 'pop', 'rock', 'jazz', 'blues',
        'metal', 'folk', 'soul', 'funk', 'house', 'techno', 'trance',
        'acoustic', 'electric', 'vocal', 'instrumental'
    ]
    
    text_lower = text.lower()
    has_english = any(word in text_lower for word in english_indicators)
    
    return has_english or ascii_chars / len(text) > 0.95

# ============================================================================
# NON-GENRE TAG FILTERING
# ============================================================================

# Tags that describe style/quality but aren't genres
STYLE_DESCRIPTORS = {
    'fast', 'slow', 'heavy', 'light', 'dark', 'bright', 'aggressive', 
    'mellow', 'upbeat', 'downtempo', 'chill', 'intense', 'powerful',
    'gentle', 'soft', 'hard', 'smooth', 'rough', 'clean', 'dirty',
    'raw', 'polished', 'melodic', 'rhythmic', 'atmospheric', 'ambient',
    'energetic', 'calm', 'peaceful', 'chaotic', 'dreamy', 'epic'
}

# Structural/technical descriptors (not genres)
TECHNICAL_DESCRIPTORS = {
    'bpm', 'beats', 'tempo', 'key', 'major', 'minor', 'chord', 'scale',
    'riff', 'riffs', 'beat', 'rhythm', 'melody', 'harmony', 'bass',
    'treble', 'low', 'high', 'mid', 'range'
}

# Instrument names (not genres)
INSTRUMENTS = {
    'guitar', 'bass', 'drums', 'piano', 'keyboard', 'synth', 'synthesizer',
    'violin', 'cello', 'trumpet', 'saxophone', 'flute', 'organ', 'accordion',
    'banjo', 'ukulele', 'harp', 'sitar', 'tabla', 'djembe', 'percussion',
    'vocals', 'voice', 'choir', 'strings', 'brass', 'woodwind'
}

# Vocal/lyrics descriptors
VOCAL_DESCRIPTORS = {
    'male', 'female', 'vocals', 'vocal', 'voice', 'singing', 'singer',
    'choir', 'acapella', 'a cappella', 'harmonies', 'lyrical', 'rap',
    'spoken', 'word', 'narrative'
}

# Production quality descriptors
PRODUCTION_DESCRIPTORS = {
    'hifi', 'lofi', 'lo-fi', 'hi-fi', 'studio', 'live', 'recorded',
    'mastered', 'mixed', 'produced', 'professional', 'amateur',
    'demo', 'rough', 'polished'
}

# Known music genres (curated list)
KNOWN_GENRES = {
    # Main genres
    'pop', 'rock', 'jazz', 'blues', 'country', 'folk', 'soul', 'funk',
    'reggae', 'ska', 'punk', 'metal', 'hip hop', 'rap', 'r&b', 'rnb',
    'electronic', 'edm', 'techno', 'house', 'trance', 'dubstep', 'dnb',
    'ambient', 'classical', 'opera', 'world', 'latin', 'salsa', 'bossa nova',
    'gospel', 'spiritual', 'new age', 'experimental', 'avant-garde',
    
    # Electronic subgenres
    'drum and bass', 'jungle', 'breakbeat', 'garage', 'hardstyle',
    'hardcore', 'electro', 'synthwave', 'vaporwave', 'chillwave',
    'idm', 'glitch', 'downtempo', 'trip hop', 'industrial',
    
    # Rock subgenres
    'alternative', 'indie', 'grunge', 'psychedelic', 'progressive',
    'hard rock', 'soft rock', 'art rock', 'post-rock', 'shoegaze',
    'noise rock', 'surf rock', 'garage rock',
    
    # Metal subgenres
    'heavy metal', 'death metal', 'black metal', 'thrash metal',
    'doom metal', 'power metal', 'nu metal', 'metalcore', 'deathcore',
    'djent', 'stoner metal', 'sludge metal',
    
    # Hip hop subgenres
    'trap', 'drill', 'boom bap', 'conscious rap', 'gangsta rap',
    'mumble rap', 'cloud rap', 'lofi hip hop',
    
    # Jazz subgenres
    'bebop', 'swing', 'fusion', 'smooth jazz', 'free jazz',
    'modal jazz', 'cool jazz', 'hard bop',
    
    # Other
    'disco', 'motown', 'bluegrass', 'americana', 'singer-songwriter',
    'bedroom pop', 'lo-fi', 'chillhop', 'synthpop', 'electropop',
    'dream pop', 'noise', 'drone', 'minimalist', 'cinematic',
    'soundtrack', 'score', 'anime', 'video game', 'j-pop', 'k-pop',
    'afrobeat', 'reggaeton', 'cumbia', 'tango', 'flamenco', 'fado',
    'celtic', 'nordic', 'balkan', 'klezmer', 'gypsy jazz'
}

def is_genre(tag):
    """Check if a tag is likely a genre"""
    tag_lower = tag.lower().strip()
    
    # Direct genre match
    if tag_lower in KNOWN_GENRES:
        return True
    
    # Check for genre compounds (e.g., "electronic rock")
    words = tag_lower.split()
    if len(words) == 2:
        if all(w in KNOWN_GENRES for w in words):
            return True
        if any(w in KNOWN_GENRES for w in words) and len(tag_lower) > 4:
            return True
    
    return False

def is_non_genre_descriptor(tag):
    """Check if tag is a descriptor but not a genre"""
    tag_lower = tag.lower().strip()
    
    all_descriptors = (
        STYLE_DESCRIPTORS | TECHNICAL_DESCRIPTORS | INSTRUMENTS |
        VOCAL_DESCRIPTORS | PRODUCTION_DESCRIPTORS
    )
    
    return tag_lower in all_descriptors

# ============================================================================
# TAG CLEANING PIPELINE
# ============================================================================

def clean_tag_string(tags_str):
    """
    Clean and extract genre tags from raw metadata tag string
    
    Input: "alternative metal, dark and heavy riffs, rich riffs, heavy and fast beats"
    Output: ["alternative metal"]
    """
    if not tags_str or not isinstance(tags_str, str):
        return []
    
    tags_str = tags_str.replace('\n', ' ').replace('\r', ' ')

    # Check if likely English
    if not is_likely_english(tags_str):
        return []
    
    # Split by common delimiters
    tags_str = tags_str.replace('/', ',').replace(';', ',').replace('|', ',')
    raw_tags = [t.strip() for t in tags_str.split(',')]
    
    clean_tags = []
    
    for tag in raw_tags:
        if not tag or len(tag) < 2:
            continue
        
        # Remove leading/trailing punctuation
        tag = tag.strip(string.punctuation + string.whitespace)
        
        # Skip if empty after cleaning
        if not tag:
            continue
        
        # Skip very long tags (likely descriptions, not genres)
        if len(tag) > 50:
            continue
        
        # Skip if contains numbers (likely BPM, years, etc.)
        if any(char.isdigit() for char in tag):
            continue
        
        # Check if it's a genre
        if is_genre(tag):
            clean_tags.append(tag.lower())
        # If not a known genre but also not a descriptor, keep it
        # (might be a niche genre we don't know)
        elif not is_non_genre_descriptor(tag) and len(tag) > 3:
            # Only keep if it's short and plausibly a genre
            words = tag.lower().split()
            if len(words) <= 3:  # Max 3 words for compound genres
                clean_tags.append(tag.lower())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in clean_tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    
    # Limit to top 3 most relevant genres
    return unique_tags[:3]

def format_genres_for_caption(genre_tags):
    """
    Format genre tags for inclusion in caption
    
    Input: ["electronic", "techno", "ambient"]
    Output: "electronic, techno, ambient"
    """
    if not genre_tags:
        return None
    
    # Capitalize each genre
    formatted = [tag.title() for tag in genre_tags]
    
    # Join with commas
    return ", ".join(formatted)

# ============================================================================
# TESTING / EXAMPLES
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("TAG CLEANING EXAMPLES")
    print("=" * 80)
    
    # Test cases from real Suno metadata
    test_cases = [
        ("folk song", "Simple genre"),
        ("New Orleans Trance", "Geographic + genre"),
        ("alternative metal, dark and heavy riffs, rich riffs, heavy and fast beats", "Genre + descriptors"),
        ("psychobilly, violin, rock", "Genres + instrument"),
        ("anime, slide guitar, japan music, house, techno, trance, edm, ambient, electro, electronic, bass, cantonese", "Mixed content"),
        ("Fast past, 145 bpm, Eurobeat, Eurobeat melody, 80´s, Italo Disco", "BPM + genres"),
        ("electric emo melancholic", "Style descriptors"),
        ("atmospheric haunting dark electronic, Female vocals, folk witch house", "Descriptors + genre + vocals"),
        ("русская музыка, folk", "Non-English + English"),
        ("", "Empty tags")
    ]
    
    for tags_str, description in test_cases:
        print(f"\n{description}")
        print(f"Input:  '{tags_str}'")
        
        cleaned = clean_tag_string(tags_str)
        formatted = format_genres_for_caption(cleaned)
        
        print(f"Cleaned: {cleaned}")
        print(f"Formatted: {formatted if formatted else 'None'}")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("Test complete!")