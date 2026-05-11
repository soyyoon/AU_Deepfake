# Training

Trains Stage 1 (Xception artifact) and Stage 2 (AU dual-head temporal)
on the data prepared by `deepfake-pp-prometheus.ipynb`.

## Expected data layout

```
{processed_root}/
├── {video_id}/
│   ├── frames/0000.png ... 0063.png    (256×256 aligned face)
│   ├── au_sequence.npy                 (64, 17), already /5.0 normalized
│   └── meta.json
└── dataset_metadata.csv                (video_id, label, source, split, ...)
```

`split ∈ {train, val, test}` is already assigned by the preprocessing
notebook (DFD: by target id; CelebDF-v2: official test list + 10% val).

## Install

```bash
pip install torch torchvision timm pandas pyyaml scikit-learn opencv-python tqdm
```

## Stage 1 — Xception (image artifact)

```bash
cd training
python train_stage1.py --config configs/stage1.yaml
```

- per-frame training: 8 frames sampled per video per epoch
- pos_weight auto-computed from class balance
- best checkpoint by validation AUC -> `checkpoints/stage1/best.pt`

## Stage 2 — AU dual-head

```bash
python train_stage2.py --config configs/stage2.yaml
```

- video-level training: full (64, 17) AU sequences
- dual loss: `seq_loss + 0.5 * frame_loss`
  - sequence head: attention pool over time, video-level supervision
  - frame head: per-frame logit, supervised with label broadcast
- best by sequence-head AUC -> `checkpoints/stage2/best.pt`

## Cascade evaluation

After both stages are trained:

```bash
python eval.py \
  --stage1-ckpt checkpoints/stage1/best.pt \
  --stage2-ckpt checkpoints/stage2/best.pt \
  --metadata-csv /kaggle/working/processed/dataset_metadata.csv \
  --processed-root /kaggle/working/processed \
  --split test \
  --low 0.4 --high 0.6 --w1 0.4 --w2 0.6
```

Reports per-stage baselines and cascade combined metric, plus the
fraction of samples that get routed to Stage 2. Use this to tune
the uncertainty band — narrower band = fewer Stage 2 calls but more
risk of low-confidence Stage 1 decisions; wider band = better accuracy
but higher latency.

## Deploy

Once you're happy with the metrics:

```bash
cp checkpoints/stage1/best.pt ../server/weights/artifact_xception.pt
cp checkpoints/stage2/best.pt ../server/weights/au_dual_head.pt
# restart server, dummy mode flag flips off automatically
```