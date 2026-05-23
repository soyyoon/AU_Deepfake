import os
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = Path("data/processed")

OUTPUT_ROOT = Path("dataset")

CSV_PATH = OUTPUT_ROOT / "metadata.csv"

SHARD_DIR = OUTPUT_ROOT / "shards"

SHARD_SIZE = 1000


# =========================================================
# OUTPUT DIR
# =========================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# STEP 1
# CREATE CSV
# =========================================================

print("=" * 60)
print("STEP 1: CREATE CSV")
print("=" * 60)

rows = []

sample_dirs = sorted(os.listdir(DATA_ROOT))

for folder in tqdm(sample_dirs):

    sample_dir = DATA_ROOT / folder

    if not sample_dir.is_dir():
        continue

    npy_path = sample_dir / "au_sequence.npy"
    meta_path = sample_dir / "meta.json"

    if not npy_path.exists():
        continue

    meta = {}

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # -----------------------------------------------------
    # emotion parsing example
    # -----------------------------------------------------

    folder_lower = folder.lower()

    emotion = ""

    emotion_candidates = [
        "angry",
        "happy",
        "sad",
        "neutral",
        "fear",
        "surprise",
        "disgust"
    ]

    for emo in emotion_candidates:
        if emo in folder_lower:
            emotion = emo
            break

    rows.append({
        "sample_id": folder,
        "npy_path": str(npy_path),
        "meta_path": str(meta_path),
        "emotion": emotion,
        "text": meta.get("text", ""),
        "label": meta.get("label", -1),
    })

df = pd.DataFrame(rows)

print(f"\nTOTAL SAMPLES: {len(df)}")

df.to_csv(CSV_PATH, index=False)

print(f"\nCSV SAVED:")
print(CSV_PATH)


# =========================================================
# STEP 2
# CREATE TAR SHARDS
# =========================================================

print("\n" + "=" * 60)
print("STEP 2: CREATE TAR SHARDS")
print("=" * 60)

num_shards = (len(df) + SHARD_SIZE - 1) // SHARD_SIZE

print(f"\nTOTAL SHARDS: {num_shards}")


for shard_idx in tqdm(range(num_shards)):

    start_idx = shard_idx * SHARD_SIZE
    end_idx   = min((shard_idx + 1) * SHARD_SIZE, len(df))

    shard_df = df.iloc[start_idx:end_idx]

    tar_name = f"dataset-{shard_idx:06d}.tar"
    tar_path = SHARD_DIR / tar_name

    with tarfile.open(tar_path, "w") as tar:

        for local_idx, row in enumerate(shard_df.itertuples()):

            key = f"{local_idx:06d}"

            # =====================================================
            # LOAD NPY
            # =====================================================

            arr = np.load(row.npy_path)

            npy_buffer = io.BytesIO()

            np.save(npy_buffer, arr)

            npy_buffer.seek(0)

            npy_info = tarfile.TarInfo(name=f"{key}.npy")

            npy_info.size = len(npy_buffer.getbuffer())

            tar.addfile(npy_info, npy_buffer)

            # =====================================================
            # META JSON
            # =====================================================

            meta = {
                "sample_id": row.sample_id,
                "emotion": row.emotion,
                "text": row.text,
                "label": int(row.label),
            }

            json_bytes = json.dumps(
                meta,
                ensure_ascii=False
            ).encode("utf-8")

            json_buffer = io.BytesIO(json_bytes)

            json_info = tarfile.TarInfo(name=f"{key}.json")

            json_info.size = len(json_bytes)

            tar.addfile(json_info, json_buffer)

    print(f"SAVED: {tar_path}")


# =========================================================
# DONE
# =========================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("""
FINAL STRUCTURE

dataset/
│
├── metadata.csv
│
└── shards/
    ├── dataset-000000.tar
    ├── dataset-000001.tar
    ├── dataset-000002.tar
    └── ...
""")