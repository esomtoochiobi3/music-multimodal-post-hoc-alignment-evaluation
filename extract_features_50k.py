import torch
import sys
import librosa
import numpy as np
import onnxruntime as ort
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--job-id', type=int, required=True)
parser.add_argument('--total-jobs', type=int, required=True)
args = parser.parse_args()

# Add paths for models
sys.path.insert(0, '/scratch/user/esomtoochiobi/thesis_work/models/myna_hybrid')
sys.path.insert(0, '/scratch/user/esomtoochiobi/thesis_work/models/MusicEmotionDetection')

from myna import Myna
from model_torch import Audio2EmotionModel

print("="*50)
print(f"FEATURE EXTRACTION - Job {args.job_id}/{args.total_jobs}")
print("="*50)

# Paths
AUDIO_DIR = Path('/scratch/user/esomtoochiobi/thesis_work/data/audio_50k')
OUTPUT_DIR = Path('/scratch/user/esomtoochiobi/thesis_work/outputs/features_50k')
EMBEDDINGS_DIR = OUTPUT_DIR / 'myna_embeddings'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

print(f"\nInput: {AUDIO_DIR}")
print(f"Output: {OUTPUT_DIR}")

# Load models
print("\nLoading models...")

# 1. MYNA
myna = Myna.from_pretrained(
    '/scratch/user/esomtoochiobi/thesis_work/models/myna_hybrid',
    trust_remote_code=True
)
myna.eval()
print("✓ MYNA loaded")

# 2. KeyMyna
key_session = ort.InferenceSession(
    '/scratch/user/esomtoochiobi/thesis_work/models/key_detection/keymyna-bb.onnx'
)
print("✓ KeyMyna loaded")

# 3. Luo Emotion Model
emotion_model = Audio2EmotionModel()
emotion_checkpoint = torch.load(
    '/scratch/user/esomtoochiobi/thesis_work/models/MusicEmotionDetection/weights/best.pth',
    map_location='cpu',
    weights_only=False
)
emotion_model.load_state_dict(emotion_checkpoint)
emotion_model.eval()
print("✓ Luo emotion model loaded")

# Key mapping
KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
        'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']

EMOTION_NAMES = ['valence', 'energy', 'tension', 'anger', 'fear', 'happy', 'sad', 'tender']

# Get audio files and split them
all_audio_files = sorted(list(AUDIO_DIR.glob('*.mp3')))
total_files = len(all_audio_files)

# Calculate this job's chunk
chunk_size = (total_files + args.total_jobs - 1) // args.total_jobs  # Ceiling division
start_idx = args.job_id * chunk_size
end_idx = min(start_idx + chunk_size, total_files)

audio_files = all_audio_files[start_idx:end_idx]

print(f"\nTotal files: {total_files}")
print(f"This job processing: {len(audio_files)} files (indices {start_idx}-{end_idx})")

# Storage for results
results = []

print("\nProcessing tracks...")
for audio_path in tqdm(audio_files):
    track_id = audio_path.stem
    
    try:
        # ===== 1. EXTRACT MYNA EMBEDDING =====
        with torch.no_grad():
            myna_output = myna.from_file(str(audio_path))
            myna_emb = myna_output.mean(dim=0).cpu().numpy()  # (768,)
        
        # Save embedding
        np.save(EMBEDDINGS_DIR / f"{track_id}.npy", myna_emb)
        
        # ===== 2. EXTRACT KEY =====
        y_key, sr_key = librosa.load(str(audio_path), sr=16000, mono=True)
        waveform = y_key.reshape(1, -1).astype(np.float32)
        key_probs = key_session.run(None, {'waveform': waveform})[0]
        predicted_key_idx = np.argmax(key_probs)
        predicted_key = KEYS[predicted_key_idx]
        key_confidence = float(key_probs[predicted_key_idx])
        
        # ===== 3. EXTRACT TEMPO =====
        y_tempo, sr_tempo = librosa.load(str(audio_path), sr=22050, duration=30.0)
        tempo, _ = librosa.beat.beat_track(y=y_tempo, sr=sr_tempo)
        tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        # ===== 4. EXTRACT EMOTIONS =====
        y_emotion, sr_emotion = librosa.load(str(audio_path), sr=22050, duration=10.0)
        
        hop_length = int(sr_emotion / 31.25)
        spec = librosa.feature.melspectrogram(
            y=y_emotion,
            sr=sr_emotion,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=149
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        spec_tensor = torch.FloatTensor(spec_db).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            emotions = emotion_model(spec_tensor).squeeze().cpu().numpy()
        
        # ===== 5. STORE RESULTS =====
        result = {
            'track_id': track_id,
            'key': predicted_key,
            'key_confidence': key_confidence,
            'tempo': tempo,
        }
        
        for name, val in zip(EMOTION_NAMES, emotions):
            result[name] = float(val)
        
        results.append(result)
        
    except Exception as e:
        print(f"\n⚠ Error processing {track_id}: {e}")
        continue

# Save results to CSV (with job ID in filename)
df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / f'features_job{args.job_id}.csv', index=False)

print(f"\n{'='*50}")
print(f"✅ Job {args.job_id} COMPLETE!")
print(f"Processed: {len(results)}/{len(audio_files)} tracks")
print(f"Output: features_job{args.job_id}.csv")
print(f"{'='*50}")