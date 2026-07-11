# Behavioral (AU-based) Deepfake Detection

Classifies a face video as **real** vs **deepfake** from a per-frame **facial
behavior time series**. The goal is detecting deepfakes encountered in the wild
(web media). v1 used OpenFace AU-intensities only and hit a data ceiling; **v2
re-preprocesses to a richer Py-Feat representation** and is the current state.

**Representation (v2):** `[64 frames × 30]` =
**20 Action Units + 3 head-pose (Pitch/Roll/Yaw) + 7 emotions**, per video.

➡️ **Full results, ablations, and generalization analysis: [RESULTS.md](RESULTS.md).**

## TL;DR

- Rich 30ch beats AU-only 20ch (**0.673 vs 0.594** ROC-AUC, xgboost) and beats the
  old OpenFace 17ch (v1 **0.549**) on the honest identity-disjoint test → the AU-
  representation bottleneck diagnosed in v1 is real and **now lifted**.
- More of the *same* data plateaus (~0.65). Unseen forgery *method* generalizes well
  (0.72, cost −0.04). The real wall is **unseen domain/dataset** (→ DFDC, 0.55).
  For in-the-wild use, **domain diversity** matters more than method count or raw size.

## Environment

Everything (extraction **and** training) runs in the **`pyfeat`** conda env:
`/home/soyoon/anaconda3/envs/pyfeat/bin/python` — Py-Feat 0.6.2, torch+CUDA, numpy,
pandas, pyyaml, matplotlib, and a **working xgboost** (unlike the old `flow_ct` env
whose scipy was ABI-broken). GPUs: 4× RTX A6000 (48 GB). Run with `PYTHONNOUSERSITE=1`
to avoid a broken user-site; the kaggle CLI lives in user-site so list/download steps
run *without* that flag (see scripts).

```bash
PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
```

## Data (v2)

Sources, chosen to be **downloadable without forms/EULA** (Kaggle one-click):
- **FaceForensics++ c23** (`xdxd003/ff-c23`) — 1,000 real + 6 forgery methods × 1,000
- **Celeb-DF v2** (`reubensuju/celeb-df-v2`) — 890 real + 5,639 synthesis
- **DFDC frame-crop sample** (`ashifurrahman34/dfdc-dataset`) — 76 real + 305 fake,
  used only as a *cross-domain* probe (new actors/methods).

Per video: `data/<id>/features.npy` `[64,30]`, `au_sequence.npy` `[64,20]` (AU-only,
for ablation), `meta.json`. Metadata: `data/dataset_metadata_v2.csv` (rebuilt from the
per-video meta.json by `finalize_metadata.py`). The experiments use a **class-balanced
subset** `data/dataset_metadata_v2_balanced.csv` — 3,782 videos (1,890 real / 1,892
fake), all 6 FF++ methods + CelebDF synthesis represented.

### Extraction pipeline (disk-constrained, resumable)

Py-Feat's AU stage is CPU-bound (GPU ~1%), so we stream **one video at a time** out of
Kaggle (stdlib `urllib`, fully URL-encoded per-file download — no big zip ever stored),
detect, save tiny `.npy`, delete the video. Auto `skip_frames` targets ~64 detected
frames (~3× faster, no info loss after resample). Sharded across workers/GPUs;
finished videos are skipped on re-run.

```bash
# list+label (kaggle lib, NO PYTHONNOUSERSITE) -> per-file stream extract (PYTHONNOUSERSITE=1)
bash scripts/run_ffpp_celebdf.sh            # FF++ c23 + Celeb-DF v2  (SUBSET=1 balanced; NGPUS=2 to limit GPUs)
$PY scripts/finalize_metadata.py            # rebuild dataset_metadata_v2.csv from meta.json
$PY scripts/extract_dfdc_frames.py          # DFDC frame-crop sample -> 30ch
```

Components: `build_filelist.py` (list+path-labels) → `make_subset.py` (class/method
balance) → `stream_extract.py` (per-file download + Py-Feat) → `finalize_metadata.py`.
`extract_features.py` is the raw-video-dir variant (e.g. DFDC-style metadata.json).

## Identity-disjoint split (the important part)

No identity may appear in two splits. `data/identity.py` parses identity tokens from
`video_id` and unions identities that co-occur in any video (faceswap/reenactment fakes
link two identities) via union-find → connected components; `data/splits.py` assigns
whole components per source with a greedy least-filled-by-ratio bin-packer (seed 42),
then **asserts** no identity spans two splits and every split has both classes and all
sources. v2 id formats parsed: `ffpp_<method>_<core>`, `celebdf_<method>_<core>`,
`dfdc_<label>_<hash>` (DFDC has no public actor ids → split is video-level).

For the balanced v2 set (`outputs/v2/splits.csv`): 1,387 identities, **0 leaks**,
realized train/val/test = 78.3 / 11.0 / 10.7 % (whole-component assignment drifts off
80/10/10), both classes and both sources in every split.

## Models

- **Main** (`models/gru.py`): 2-layer BiGRU (hidden 128) → temporal attention pooling
  → MLP → 1 logit. AdamW(1e-3), batch 128, ≤60 epochs, early stop on val AUC (patience
  8), `BCEWithLogitsLoss(pos_weight)`, val-F1 threshold. ~485k params.
- **Alt** (`models/transformer.py`): small Transformer encoder, same interface.
- **Baseline** (`baseline.py`): per-channel stats (mean/std/min/max/range + Δ mean/std)
  → **xgboost** (available in this env) — the strongest model here.
- Input z-scored per channel using **train-split** statistics. `feature_file` in the
  config selects 30ch (`features.npy`) vs 20ch (`au_sequence.npy`).

Metrics (`metrics.py`, pure numpy): ROC-AUC, PR-AUC, F1, balanced-acc, EER, confusion,
val-driven threshold; evaluation decomposed **by source** (FFpp / CelebDF).

## Run the experiments

```bash
$PY scripts/qc_features.py --balanced       # feature quality check -> outputs/qc/
bash scripts/run_v2_experiment.sh           # 30ch vs 20ch: train+eval+baseline + comparison
$PY scripts/learning_curve.py               # data-scaling curve  -> outputs/v2/learning_curve.png
$PY scripts/method_holdout.py               # unseen forgery-method generalization (FF++ LOMO)
$PY scripts/diversity_experiment.py         # cross-dataset (DFDC) gap
# smoke: EPOCHS=2 bash scripts/run_v2_experiment.sh
```

Configs: `configs/v2_features.yaml` (30ch), `configs/v2_au.yaml` (20ch) — identical
except `feature_file`/`n_channels`/output dir, sharing `outputs/v2/splits.csv`.
Outputs land under `outputs/v2/{features,au}/` (`metrics*.json`, checkpoints, plots).

## Results (identity-disjoint test) — see [RESULTS.md](RESULTS.md) for full tables

| representation | xgboost | BiGRU | FFpp | CelebDF |
|---|--:|--:|--:|--:|
| **RICH 30ch** (AU+pose+emotion) | **0.673** | 0.590 | 0.754 | 0.587 |
| AU-only 20ch | 0.594 | 0.571 | 0.706 | 0.471 |
| *v1 OpenFace 17ch (old)* | — | *0.549* | — | — |

**Generalization** (xgboost 30ch): unseen-identity 0.76 → unseen-**method** 0.72
(cost −0.04) → unseen-**domain**/DFDC 0.55 (cost −0.17). Behavioral features transfer
across forgery techniques; **domain shift is the bottleneck** for in-the-wild detection.

## v1 (historical)

v1 trained on CelebDF + DFD with OpenFace `[64,17]` AU intensities; identity-disjoint
test ROC-AUC **0.549** (leaky split 0.654). Confirmatory experiments showed the model
memorized identity (train AUC ~0.998) while no AU-feature engineering moved test AUC —
i.e. the *representation*, not the model, was the bottleneck. v2 confirms this and
lifts the ceiling. The old `data/dataset_metadata.csv` (v1 labels) is kept for record;
the old `data/vid_*/au_sequence.npy` dirs were deleted (superseded, 154 MB).
