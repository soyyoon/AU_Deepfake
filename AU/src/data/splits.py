"""identity-disjoint train/val/test re-split.

Strategy: build connected-component groups (data/identity.py), then assign whole
components to splits per source with a greedy "least-filled-by-ratio" bin-packer
targeting 0.8/0.1/0.1. Because faceswap chains create one component spanning ~70%
of CelebDF, exact 0.8 is impossible without splitting that component (which would
reintroduce identity leakage); we keep components whole, so realized ratios drift
to ~0.77/0.12/0.11. Leak-free evaluation is prioritized over exact ratios
(success criterion in the plan).

Every identity (hence every component) lands in exactly one split -> no identity
appears in two splits (asserted).
"""
import csv
import collections
import random

from .identity import build_groups, extract_identities

SPLITS = ("train", "val", "test")
TARGET = {"train": 0.8, "val": 0.1, "test": 0.1}


def assign_components(comp_sizes, source_total, seed):
    """Greedy: largest components first, each to the split most under its quota."""
    quota = {s: TARGET[s] * source_total for s in SPLITS}
    filled = {s: 0 for s in SPLITS}
    assignment = {}
    rng = random.Random(seed)
    # sort by size desc; shuffle equal-size groups for reproducible tie-breaking
    items = sorted(comp_sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    rng.shuffle(items)  # break correlation; stable size order restored next line
    items.sort(key=lambda kv: -kv[1])
    for comp, n in items:
        ratios = {s: (filled[s] / quota[s] if quota[s] > 0 else 1.0) for s in SPLITS}
        target_split = min(SPLITS, key=lambda s: ratios[s])
        assignment[comp] = target_split
        filled[target_split] += n
    return assignment


def make_splits(metadata_csv, out_csv, seed=42, verbose=True):
    rows = list(csv.DictReader(open(metadata_csv)))
    groups, _ = build_groups([(r["video_id"], r["source"]) for r in rows])

    # per-source component sizes
    src_comps = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        src_comps[r["source"]][groups[r["video_id"]]] += 1

    comp_split = {}
    for source, comps in src_comps.items():
        total = sum(comps.values())
        comp_split.update(assign_components(comps, total, seed))

    # write splits.csv
    out_rows = []
    for r in rows:
        g = groups[r["video_id"]]
        out_rows.append({
            "video_id": r["video_id"], "label": r["label"],
            "source": r["source"], "group_id": g, "split": comp_split[g],
        })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "label", "source", "group_id", "split"])
        w.writeheader()
        w.writerows(out_rows)

    _verify_and_log(out_rows, verbose)
    return out_rows


def _verify_and_log(out_rows, verbose):
    # 1) identity-disjointness: each identity token must appear in exactly one split
    ident_splits = collections.defaultdict(set)
    for r in out_rows:
        for ident in extract_identities(r["video_id"], r["source"]):
            ident_splits[ident].add(r["split"])
    leaked = {i: s for i, s in ident_splits.items() if len(s) > 1}
    assert not leaked, f"LEAKAGE: identities in multiple splits: {list(leaked)[:5]}"

    # 2) each group in exactly one split
    grp_splits = collections.defaultdict(set)
    for r in out_rows:
        grp_splits[r["group_id"]].add(r["split"])
    assert all(len(s) == 1 for s in grp_splits.values()), "group spans splits"

    if not verbose:
        return
    n = len(out_rows)
    all_sources = sorted({r["source"] for r in out_rows})   # source-agnostic (v1: CelebDF/DFD; v2: FFpp/CelebDF)
    print(f"[splits] total {n} videos, {len(ident_splits)} identities, "
          f"sources={all_sources}, identity-disjoint assert PASSED (0 leaks)")
    for split in SPLITS:
        sub = [r for r in out_rows if r["split"] == split]
        lab = collections.Counter(r["label"] for r in sub)
        src = collections.Counter(r["source"] for r in sub)
        src_str = " ".join(f"{s}={src.get(s,0)}" for s in all_sources)
        print(f"  {split:5s} n={len(sub):5d} ({len(sub)/n:5.1%})  "
              f"real(0)={lab.get('0',0):4d} fake(1)={lab.get('1',0):4d}  {src_str}")
        assert lab.get("0", 0) > 0 and lab.get("1", 0) > 0, f"{split} missing a class"
        for s in all_sources:
            assert src.get(s, 0) > 0, f"{split} missing source {s}"
    print("  per-split class & source presence assert PASSED")
