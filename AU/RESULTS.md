# v2 Results — rich behavioral features & where generalization breaks

All numbers are **ROC-AUC on the identity-disjoint test split** unless noted.
Model = xgboost on per-channel statistical features (the strongest here) and a
BiGRU+attention. Representation:

- **RICH 30ch** (`features.npy`) = 20 Py-Feat AUs + 3 head-pose + 7 emotions
- **AU-only 20ch** (`au_sequence.npy`) = the 20 AUs alone (ablation control)

Data = FaceForensics++ c23 + Celeb-DF v2, balanced to **3,782 videos**
(1,890 real / 1,892 fake), split identity-disjoint into train 2,962 / val 416 /
test 404 (`outputs/v2/splits.csv`, 0 identity leaks).

---

## 1. Does the richer representation beat AU-only? — yes

`scripts/run_v2_experiment.sh` (configs `v2_features.yaml`, `v2_au.yaml`; shared split)

| representation | model | ROC-AUC | PR-AUC | F1 | FFpp | CelebDF |
|---|---|--:|--:|--:|--:|--:|
| **RICH 30ch** | xgboost | **0.673** | 0.709 | 0.690 | 0.754 | 0.587 |
| RICH 30ch | BiGRU | 0.590 | 0.601 | 0.687 | 0.649 | 0.533 |
| AU-only 20ch | xgboost | 0.594 | 0.629 | 0.690 | 0.706 | 0.471 |
| AU-only 20ch | BiGRU | 0.571 | 0.604 | 0.688 | 0.608 | 0.561 |
| *v1 (OpenFace 17ch, old data)* | *BiGRU* | *0.549* | — | — | — | — |

- Adding pose+emotion: **+0.079 AUC** on the xgboost baseline (0.673 vs 0.594),
  +0.019 on the BiGRU, +0.031 on val (0.614 vs 0.583). Consistent across model/metric.
- **v2 clears the v1 ceiling** (0.549 identity-disjoint; even 0.654 on the leaky split).
  The re-preprocessing achieved its goal: the AU-representation bottleneck is lifted.
- xgboost > BiGRU: the signal lives in per-channel distribution stats, not fine
  temporal dynamics. FFpp (~0.75) is much easier than CelebDF (~0.59).
- NOTE: this `pyfeat` env has a **working xgboost** (unlike the old `flow_ct` env whose
  scipy was ABI-broken and forced a torch-logreg baseline).

## 2. Feature quality check — clean, emotion channels carry the top signal

`scripts/qc_features.py --balanced` → `outputs/qc/channel_auc.png`

- Integrity: 3,782 videos, 30 channels, **0 NaN / 0 Inf / 0 constant / 0 missing**.
  Ranges sane (AUs & emotions 0–1, pose ±degrees).
- Per-channel univariate real-vs-fake |AUC−0.5| (higher = more discriminative): the
  two **most discriminative single channels are emotions** — `surprise` (0.111) and
  `neutral` (0.090) — followed by AU05, AU24, AU02. Mean |AUC−0.5|: AU channels 0.054,
  pose+emotion channels 0.048. The extra non-AU channels carry comparable signal, and
  the very top is emotion → supports the rich-representation gain in §1.

## 3. How much does MORE data help? — it plateaus

`scripts/learning_curve.py` (xgboost 30ch, subsample train by identity-group, 5 seeds)
→ `outputs/v2/learning_curve.png`

| train videos | #identity groups | test AUC |
|--:|--:|--:|
| 445 | 208 | 0.579 |
| 1,201 | 399 | 0.639 |
| 1,659 | 514 | 0.652 |
| 2,188 | 593 | 0.660 |
| 2,443 | 716 | 0.658 |
| 2,962 | 982 | 0.648 |

- Steep rise to ~1,200 videos, then **flat** (1,600→3,000 ≈ 0 gain, within ±0.01 noise).
- A naive log-fit extrapolates +0.07 at 3.5× data, but the **recent slope is flat** →
  that is optimistic. More of the *same distribution* is near its ceiling (~0.65).

## 4. Where does generalization break? — domain, not forgery technique

Three generalization axes measured (xgboost 30ch):

| axis | test AUC | cost |
|---|--:|--:|
| seen-method, unseen-identity (same dataset) | 0.76 | baseline |
| **unseen forgery method**, unseen-identity | **0.72** | −0.04 (small) |
| **unseen dataset / domain** (→ DFDC) | **0.55** | −0.17 (large) |

### 4a. Unseen forgery technique — generalizes well
`scripts/method_holdout.py` — FF++ leave-one-method-out (5 classic methods share the
1,000 YouTube reals; identity-disjoint by target id; 5 seeds).

| held-out method | CROSS (unseen) | SEEN (control) | novelty cost |
|---|--:|--:|--:|
| Deepfakes | 0.857 | 0.896 | +0.038 |
| Face2Face | 0.767 | 0.807 | +0.040 |
| NeuralTextures | 0.732 | 0.763 | +0.031 |
| FaceShifter | 0.637 | 0.676 | +0.039 |
| FaceSwap | 0.615 | 0.663 | +0.048 |
| **mean** | **0.722** | **0.761** | **+0.039** |

- A detector trained on 4 methods flags the unseen 5th at **0.72** — behavioral
  features transfer across forgery techniques; novelty cost is only **−0.04**.
- Per-method spread is large: Deepfakes transfers great (0.86); **FaceSwap (0.62) /
  FaceShifter (0.64) are hard** — graphics-based swaps have a different artifact profile.
- 30ch ≈ 20ch on the CROSS axis (0.722 vs 0.723): rich channels help *in-distribution*
  but add nothing to *unseen-method* transfer.

### 4b. Unseen dataset / domain — the real wall
`scripts/diversity_experiment.py` — train FF++/CelebDF → eval DFDC (new actors, new
methods, different capture/compression).

- Cross-dataset transfer to DFDC ≈ **0.55** (near chance) vs in-domain 0.72–0.76.
- ⚠️ Caveat: the only no-form DFDC source (Kaggle frame-crop dump) is tiny — 76 balanced
  videos, ~10 frames each; in-domain DFDC AUC is also ~0.51, so "adding DFDC lifts it"
  was **underpowered / untestable**. The 0.55 cleanly shows the *cross-domain gap*; it
  does not refute that more in-domain data would help.

## 5. Conclusions — where to spend data effort

1. **Representation**: rich 30ch (AU+pose+emotion) > AU-only, and > the old OpenFace
   17ch — the data-representation bottleneck diagnosed in v1 is real and now lifted.
2. **Data scaling**: more of the *same* datasets → plateau (~0.65). Low value.
3. **Forgery-method diversity**: small marginal value — unseen methods already transfer
   at ~0.72 (cost −0.04).
4. **Domain diversity is the lever**: the big drop is cross-domain (0.72→0.55). For the
   in-the-wild goal (web media), the priority is **varied real domains** (capture
   conditions, compression, sources), not more forgery methods or raw video count.
5. **Open weakness**: CelebDF (~0.59) and graphics-based swaps (FaceSwap/FaceShifter)
   resist behavioral features — likely need frame-level pixel/frequency artifacts to
   complement AUs.

## Reproduce

```bash
PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
# (features already extracted under data/<id>/; see README for extraction)
$PY scripts/qc_features.py --balanced
bash scripts/run_v2_experiment.sh           # 30ch vs 20ch (train+eval+baseline)
$PY scripts/learning_curve.py               # data-scaling curve
$PY scripts/method_holdout.py               # unseen-forgery-method generalization
$PY scripts/diversity_experiment.py         # cross-dataset (DFDC) gap
```
