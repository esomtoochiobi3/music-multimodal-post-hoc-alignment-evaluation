import json
import random

# read all metadata
print("Reading metadata...")
tracks = []
with open('/scratch/user/esomtoochiobi/thesis_work/data/suno_660k/suno.jsonl', 'r') as f:
    for line in f:
        tracks.append(json.loads(line))

print(f"Total tracks: {len(tracks)}")

# randomly sample 50000
random.seed(42)  # for reproducibility
sample = random.sample(tracks, 50000)

# save sample
with open('/scratch/user/esomtoochiobi/thesis_work/data/sample_50k.jsonl', 'w') as f:
    for track in sample:
        f.write(json.dumps(track) + '\n')

print(f"✓ Saved 50k random tracks to sample_50k.jsonl")

# also save just the IDs and URLs for easy downloading
with open('/scratch/user/esomtoochiobi/thesis_work/data/sample_50k_urls.txt', 'w') as f:
    for track in sample:
        f.write(f"{track['id']}\t{track['audio_url']}\n")

print(f"✓ Saved URLs to sample_50k_urls.txt")