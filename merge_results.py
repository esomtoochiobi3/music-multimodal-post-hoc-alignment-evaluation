import pandas as pd
from pathlib import Path

output_dir = Path('/scratch/user/esomtoochiobi/thesis_work/outputs/features_50k')

# Find all job result files
csv_files = sorted(output_dir.glob('features_job*.csv'))

print(f"Found {len(csv_files)} result files")

# Merge them
dfs = [pd.read_csv(f) for f in csv_files]
merged = pd.concat(dfs, ignore_index=True)

# Save combined
merged.to_csv(output_dir / 'features_50k.csv', index=False)

print(f"✓ Merged {len(merged)} total tracks")
print(f"Saved to: {output_dir}/features_50k.csv")