import math
import tarfile
from pathlib import Path
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = Path("data/processed")

OUTPUT_DIR = Path("au_dataset_chunks")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SHARDS = 11


# =========================================================
# SAMPLE FOLDERS
# =========================================================

sample_dirs = sorted([
    p for p in DATA_ROOT.iterdir()
    if p.is_dir()
])

total_samples = len(sample_dirs)

print(f"TOTAL SAMPLE DIRS: {total_samples}")

samples_per_shard = math.ceil(total_samples / NUM_SHARDS)

print(f"SAMPLES PER SHARD: {samples_per_shard}")


# =========================================================
# CREATE TAR SHARDS
# =========================================================

for shard_idx in range(NUM_SHARDS):

    start_idx = shard_idx * samples_per_shard
    end_idx = min((shard_idx + 1) * samples_per_shard, total_samples)

    shard_samples = sample_dirs[start_idx:end_idx]

    if len(shard_samples) == 0:
        continue

    tar_path = OUTPUT_DIR / f"au_dataset_chunk_{shard_idx:03d}.tar"

    print(f"\nCREATING: {tar_path}")
    print(f"SAMPLES: {len(shard_samples)}")

    with tarfile.open(tar_path, "w") as tar:

        for sample_dir in tqdm(shard_samples):

            tar.add(
                sample_dir,
                arcname=sample_dir.name
            )

    print(f"SAVED: {tar_path}")


print("\nDONE")