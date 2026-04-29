# When Do Theory-Enriched Captions Help Music Retrieval?

### A Controlled Study of Text Encoder Pretraining in Post-Hoc Alignment

**ISMIR 2026 Submission**

This repository contains the code, caption datasets, and figures for our ISMIR 2026 paper. We study whether enriching music captions with structural (key, tempo) and affective (emotion) attributes improves audio-text retrieval under a post-hoc alignment framework, and whether any benefit depends on the choice of text encoder.

---

## Overview

We train lightweight MLP adapters that project frozen [MYNA-Hybrid](https://arxiv.org/abs/2502.12511) audio embeddings and frozen text embeddings (CLAP or MuQ-MuLan) into a shared 256-dimensional retrieval space. We evaluate four caption variants across 1k, 10k, and 50k tracks from the [Suno-660k](https://huggingface.co/datasets/nyuuzyou/suno) dataset:

| Variant               | Example                                              |
| --------------------- | ---------------------------------------------------- |
| Genre-only (baseline) | _rock_                                               |
| Structural            | _fast-paced rock in E minor_                         |
| Affective             | _energetic rock with a dark, tense mood_             |
| Full                  | _fast-paced rock in E minor with a dark, tense mood_ |

**Key finding:** Under CLAP, genre-only captions outperform theory-enriched variants. Replacing CLAP with the music-specific MuQ-MuLan encoder reverses this ordering — full captions become the best condition, with mAP improving from 0.038 to 0.100.

---

## Repository Structure

```
├── README.md
├── clean_tags.py                  # Clean and normalize genre tags
├── sample_tracks.py               # Sample tracks from Suno-660k
├── download_audio_50k.py          # Download audio files
├── extract_features_50k.py        # Extract MYNA audio embeddings
├── discretize_features.py         # Discretize tempo and emotion features
├── generate_captions_50k.py       # Generate templated captions (50k)
├── generate_captions_v2.py        # Generate naturalized captions
├── encode_captions_clap.py        # Encode captions with CLAP
├── merge_results.py               # Merge and aggregate results
├── dataset.py                     # Dataset and data loading utilities
├── models.py                      # Adapter architecture
├── train_adapters.py              # Main training script
├── captions/
│   ├── captions_1k/               # 1k scale captions (templated + naturalized)
│   ├── captions_10k/              # 10k scale captions (templated + naturalized)
│   └── captions_50k/              # 50k scale captions (templated + naturalized)
└── figures/
    ├── ismir2026_pipeline.png     # Architecture diagram
    ├── umap_comparison_50k.pdf    # UMAP embedding space analysis
    ├── cosine_sim_50k.pdf         # Cosine similarity distributions
    ├── UMAP/                      # Additional UMAP plots
    └── t-SNE/                     # t-SNE plots
```

---

## Setup

```bash
git clone https://github.com/esomtoochiobi3/music-multimodal-post-hoc-alignment-evaluation
cd music-multimodal-post-hoc-alignment-evaluation
pip install torch torchaudio transformers librosa numpy pandas scikit-learn
```

**Pretrained models required:**

- MYNA-Hybrid audio encoder — see [arXiv:2502.12511](https://arxiv.org/abs/2502.12511) for weights
- CLAP text encoder — `laion/clap-htsat-unfused` from HuggingFace
- KeyMyna key detection probe — from the MYNA repository
- A2E emotion model — [jeffreyzluo/MusicEmotionDetection](https://github.com/jeffreyzluo/MusicEmotionDetection)

---

## Reproducing the Pipeline

### 1. Sample and download audio

```bash
python sample_tracks.py --n 50000 --seed 42
python download_audio_50k.py
```

### 2. Extract features

```bash
python extract_features_50k.py --audio_dir /path/to/audio --output_dir /path/to/embeddings
python discretize_features.py
```

### 3. Generate captions

```bash
# Templated captions
python generate_captions_50k.py --features_dir /path/to/features --output_dir captions/captions_50k

# Naturalized captions (requires Claude API key)
python generate_captions_v2.py --input_dir captions/captions_50k --output_dir captions/captions_50k/naturalized
```

### 4. Encode captions

```bash
python encode_captions_clap.py --captions_dir captions/captions_50k --output_dir /path/to/text_embeddings
```

### 5. Train adapters

```bash
python train_adapters.py \
  --audio_embedding_dir /path/to/audio_embeddings \
  --text_embedding_dir /path/to/text_embeddings \
  --output_dir /path/to/results \
  --audio_dim 768 \
  --text_dim 512 \
  --hidden_dim 512 \
  --output_dim 256 \
  --batch_size 512 \
  --num_epochs 200 \
  --learning_rate 5e-4 \
  --weight_decay 1e-4 \
  --patience 20
```

---

## Caption Datasets

Pre-generated captions for all three scales are included in the `captions/` directory. Each scale contains four templated variants and four naturalized variants:

- `captions_baseline_{scale}.csv` — genre tags only
- `captions_structural_{scale}.csv` — genre + key + tempo
- `captions_affective_{scale}.csv` — genre + emotion descriptors
- `captions_full_{scale}.csv` — genre + key + tempo + emotion
- `naturalized/` — LLM-rewritten fluent versions of each variant

Audio tracks are from the [Suno-660k dataset](https://huggingface.co/datasets/nyuuzyou/suno) on HuggingFace.

---
