# Semantic text-guided continual skeleton-based action recognition

A two-phase continual learning framework for skeleton-based action recognition.
Phase 1 pretrains five Part Motion Autoencoders (PM-AE) on PKU-MMD Part II.
Phase 2 trains a Continual Motion Autoencoder (CM-AE) incrementally on NTU RGB+D,
guided by frozen shared motion features and BERT class-name semantic alignment.
A Selective Semantic Class Replay algorithm reduces forgetting using motion-description
similarity to decide which previous classes to rehearse.

---

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install numpy
pip install scikit-learn
```

Tested with Python 3.9+, PyTorch 2.x, CUDA 11.8.

---

## Project Structure

All Python scripts live in the same root folder. The `data/` folder sits
alongside them and holds all raw files, converted sequences and pre-computed
features. The paths in the code are already set to match this layout.

```
SA-CIL/                          ← root (run all scripts from here)
├── main.py                      # Single-run entry point
├── main_runs.py                 # Multi-run entry point (seeds / class-order shuffles)
├── models.py                    # PM-AE, CM-AE, CIL classifier definitions
├── dataloader.py                # NTU and PKU data loading, normalisation, task splits
├── bert_feat.py                 # BERT features, SPICE similarity, SentenceBertBank
├── txt2npy_ntu.py               # Convert NTU raw .skeleton txt files to npy
├── pku_txt2npy.py               # Convert PKU-MMD Part II txt files to npy
├── collect_results.py           # Aggregate per-run JSON result files
│
├── data/                        ← all data and pre-computed features go here
│   ├── ntu/
│   │   ├── raw60/               # Raw NTU RGB+D 60 .skeleton txt files
│   │   ├── raw120/              # Raw NTU RGB+D 120 .skeleton txt files
│   │   ├── npy60/               # Converted NTU-60 npy files (one per sequence)
│   │   ├── npy120/              # Converted NTU-120 npy files (one per sequence)
│   │   ├── NTU_RGBD_samples_with_missing_skeletons.txt
│   │   ├── NTU_RGBD120_samples_with_missing_skeletons.txt
│   │   ├── bert_feats_60.npy          # BERT class-name features — NTU-60
│   │   ├── bert_feats_120.npy         # BERT class-name features — NTU-120
│   │   ├── bert_sentence_feats_60.npy # BERT sentence features  — NTU-60
│   │   └── bert_sentence_feats_120.npy
│   └── pku/
│       ├── raw/                 # Raw PKU-MMD Part II txt skeleton files
│       ├── npy/                 # Converted PKU-MMD npy files (one per sequence)
│       └── bert_feats.npy       # BERT class-name features — PKU-MMD
│
├── checkpoints/                 # Checkpoints for single-run experiments (auto-created)
├── ckpt_run/                    # Checkpoints for multi-run experiments (auto-created)
└── results/                     # Per-run logs and JSON result files
```

---

## Data Download

### NTU RGB+D 60 and 120

Download the **skeleton-only** data from the official repository:

> https://github.com/shahroudy/NTURGB-D

Request access via the form on that page. Once approved, download:
- `nturgbd_skeletons_s001_to_s017.zip` (NTU-60)
- `nturgbd_skeletons_s018_to_s032.zip` (NTU-120 extra)

Extract `.skeleton` files into:
- `data/ntu/raw60/`   — for NTU-60 sequences
- `data/ntu/raw120/`  — for all NTU-120 sequences (60 + 120)

The missing skeleton lists are available from the same repository:
- `NTU_RGBD_samples_with_missing_skeletons.txt`
- `NTU_RGBD120_samples_with_missing_skeletons.txt`

Place them in `data/ntu/`.

### PKU-MMD Part II

Download PKU-MMD Part II skeleton data from:

> https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html

Extract the txt skeleton files into `data/pku/raw/`.

---

## Data Preprocessing

### Convert NTU txt files to npy

```bash
# For NTU-60 (adjust save_npy_path and load_txt_path inside the script)
python txt2npy_ntu.py
```

Edit the paths at the top of `txt2npy_ntu.py`:
```python
save_npy_path = './data/ntu/npy60/'    # or npy120/
load_txt_path = './data/ntu/raw60/'    # or raw120/
missing_file_path = './data/ntu/NTU_RGBD_samples_with_missing_skeletons.txt'
```

### Convert PKU-MMD Part II txt files to npy

```bash
python pku_txt2npy.py --input_dir data/pku/raw --output_dir data/pku/npy
```

---

## BERT Features

BERT class-name features and motion-description sentence features must be
generated once before training. Run:

```bash
# NTU-60
python bert_feat.py --dataset ntu --output data/ntu/bert_feats_60.npy

# NTU-120
python bert_feat.py --dataset ntu120 --output data/ntu/bert_feats_120.npy

# Sentence features (for selective replay)
python bert_feat.py --dataset ntu --sentence --output data/ntu/bert_sentence_feats_60.npy
python bert_feat.py --dataset ntu120 --sentence --output data/ntu/bert_sentence_feats_120.npy

# PKU-MMD
python bert_feat.py --dataset pku --output data/pku/bert_feats.npy
```

### LLM-generated motion descriptions

The structured natural language motion descriptions used for selective
replay similarity computation (NTU_SENTENCE_DESCRIPTIONS) are **already
provided** in `bert_feat.py`. No external generation step is needed.
They are stored in the `NTU_SENTENCE_DESCRIPTIONS` dictionary starting
at line 130 of `bert_feat.py`, one entry per NTU RGB+D action class.

---

## Running Experiments

### Single run — `main.py`

Runs one experiment on a single dataset/protocol without a random seed
or class-order variant.

```bash
# NTU-60 X-Sub
python main.py \
    --novel_sample_fraction 0.03 \
    --lambda_align 0.3 \
    --dataset ntu60 \
    --protocol xsub \
    --part_ckpt_dir checkpoints \
    --checkpoint_dir checkpoints/ntu60_xsub \
    > results/ntu60_xsub.log 2>&1

# NTU-60 X-View
python main.py \
    --novel_sample_fraction 0.03 \
    --lambda_align 0.3 \
    --dataset ntu60 \
    --protocol xview \
    --part_ckpt_dir checkpoints \
    --checkpoint_dir checkpoints/ntu60_xview \
    > results/ntu60_xview.log 2>&1

# NTU-120 X-Sub
python main.py \
    --novel_sample_fraction 0.03 \
    --lambda_align 0.3 \
    --dataset ntu120 \
    --protocol xsub \
    --part_ckpt_dir checkpoints \
    --checkpoint_dir checkpoints/ntu120_xsub \
    > results/ntu120_xsub.log 2>&1

# NTU-120 X-Set
python main.py \
    --novel_sample_fraction 0.03 \
    --lambda_align 0.3 \
    --dataset ntu120 \
    --protocol xset \
    --part_ckpt_dir checkpoints \
    --checkpoint_dir checkpoints/ntu120_xset \
    > results/ntu120_xset.log 2>&1
```

### Multiple runs — `main_runs.py`

Runs three seeds (S1/S2/S3) and three class-order shuffles (O1/O2/O3)
for each dataset/protocol combination.
The `--part_ckpt_dir` points to the shared pre-trained PM-AE checkpoints.
Each run writes its own log and a JSON result file under `results/`.

#### NTU-60 X-Sub

```bash
mkdir -p results ckpt_run

# Seeds
stdbuf -oL python main_runs.py --seed 0 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xsub_seed0 > results/ntu60_xsub_seed0.log 2>&1
stdbuf -oL python main_runs.py --seed 1 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xsub_seed1 > results/ntu60_xsub_seed1.log 2>&1
stdbuf -oL python main_runs.py --seed 2 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xsub_seed2 > results/ntu60_xsub_seed2.log 2>&1

# Class-order shuffles
stdbuf -oL python main_runs.py --shuffle_variant 0 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xsub_shuf0 > results/ntu60_xsub_shuf0.log 2>&1
stdbuf -oL python main_runs.py --shuffle_variant 1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xsub_shuf1 > results/ntu60_xsub_shuf1.log 2>&1
stdbuf -oL python main_runs.py --shuffle_variant 2 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xsub_shuf2 > results/ntu60_xsub_shuf2.log 2>&1
```

#### NTU-60 X-View

```bash
stdbuf -oL python main_runs.py --seed 0 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xview --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xview_seed0 > results/ntu60_xview_seed0.log 2>&1
stdbuf -oL python main_runs.py --seed 1 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xview --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xview_seed1 > results/ntu60_xview_seed1.log 2>&1
stdbuf -oL python main_runs.py --seed 2 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xview --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xview_seed2 > results/ntu60_xview_seed2.log 2>&1

stdbuf -oL python main_runs.py --shuffle_variant 0 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xview --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xview_shuf0 > results/ntu60_xview_shuf0.log 2>&1
stdbuf -oL python main_runs.py --shuffle_variant 1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xview --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xview_shuf1 > results/ntu60_xview_shuf1.log 2>&1
stdbuf -oL python main_runs.py --shuffle_variant 2 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu60 --protocol xview --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu60_xview_shuf2 > results/ntu60_xview_shuf2.log 2>&1
```

#### NTU-120 X-Sub

```bash
stdbuf -oL python main_runs.py --seed 0 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xsub_seed0 > results/ntu120_xsub_seed0.log 2>&1
stdbuf -oL python main_runs.py --seed 1 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xsub_seed1 > results/ntu120_xsub_seed1.log 2>&1
stdbuf -oL python main_runs.py --seed 2 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xsub_seed2 > results/ntu120_xsub_seed2.log 2>&1

stdbuf -oL python main_runs.py --shuffle_variant 0 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xsub_shuf0 > results/ntu120_xsub_shuf0.log 2>&1
stdbuf -oL python main_runs.py --shuffle_variant 1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xsub_shuf1 > results/ntu120_xsub_shuf1.log 2>&1
stdbuf -oL python main_runs.py --shuffle_variant 2 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xsub --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xsub_shuf2 > results/ntu120_xsub_shuf2.log 2>&1
```

#### NTU-120 X-Set

```bash
stdbuf -oL python main_runs.py --seed 0 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xset --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xset_seed0 > results/ntu120_xset_seed0.log 2>&1
stdbuf -oL python main_runs.py --seed 1 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xset --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xset_seed1 > results/ntu120_xset_seed1.log 2>&1
stdbuf -oL python main_runs.py --seed 2 --shuffle_variant -1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xset --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xset_seed2 > results/ntu120_xset_seed2.log 2>&1

stdbuf -oL python main_runs.py --shuffle_variant 0 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xset --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xset_shuf0 > results/ntu120_xset_shuf0.log 2>&1
stdbuf -oL python main_runs.py --shuffle_variant 1 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xset --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xset_shuf1 > results/ntu120_xset_shuf1.log 2>&1
stdbuf -oL python main_runs.py --shuffle_variant 2 --novel_sample_fraction 0.03 --lambda_align 0.3 --dataset ntu120 --protocol xset --part_ckpt_dir ckpt_run --checkpoint_dir ckpt_run/ntu120_xset_shuf2 > results/ntu120_xset_shuf2.log 2>&1
```

### Collecting results

After all runs complete, aggregate per-run JSON result files:

```bash
python collect_results.py --dataset ntu60  --protocol xsub
python collect_results.py --dataset ntu120 --protocol xset
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `ntu60` | Dataset: `ntu60` or `ntu120` |
| `--protocol` | `xview` | Protocol: `xsub`, `xview`, `xset` |
| `--lambda_align` | `0.3` | Weight of semantic alignment loss |
| `--novel_sample_fraction` | `0.03` | Fraction of samples per replayed class |
| `--part_ckpt_dir` | — | Directory containing pre-trained PM-AE checkpoints |
| `--checkpoint_dir` | — | Directory to save CM-AE and classifier checkpoints |
| `--seed` | `-1` | Random seed for Phase 2+3 (`main_runs.py` only) |
| `--shuffle_variant` | `-1` | Class-order shuffle variant 0/1/2 (`main_runs.py` only) |

---

## Citation

If you use this code, please cite the corresponding paper.
