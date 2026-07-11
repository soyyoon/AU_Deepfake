"""video_id -> identity set extraction + union-find grouping.

v1 rules (data/dataset_metadata.csv, vid_* ids):
  CelebDF real : vid_003910_id20_0003          -> {celeb_id20}
                 vid_003634_00208 (YouTube-real)-> {yt_00208}
  CelebDF fake : vid_004473_id16_id0_0010       -> {celeb_id16, celeb_id0}
  DFD real/fake: vid_..._24__... / vid_..._12_13__..HASH -> {dfd_24} / {dfd_12,dfd_13}

v2 rules (dataset_metadata_v2.csv, <dataset>_<method>_<core> ids):
  FFpp real    : ffpp_real_000                  -> {ff_000}
  FFpp fake    : ffpp_Deepfakes_000_003         -> {ff_000, ff_003}   (target_source pair)
                 ffpp_DeepFakeDetection_01_02__scene__HASH -> {ff_01, ff_02} (actor pair)
  CelebDF real : celebdf_real_id0_0000          -> {celeb_id0}
                 celebdf_real_00000 (YouTube)   -> {yt_00000}
  CelebDF fake : celebdf_synthesis_id0_id1_0000 -> {celeb_id0, celeb_id1}

faceswap/reenactment fakes link two identities; identities that co-occur in any
video are merged into one connected component (union-find) so no identity leaks
across train/val/test.
"""
import re

_VID_PREFIX = re.compile(r"^vid_\d+_")
_V2_PREFIX = re.compile(r"^(?:ffpp|celebdf|dfdc)_[A-Za-z0-9]+_")  # <dataset>_<method>_
_CELEB_ID = re.compile(r"id\d+")
_LEADING_NUM = re.compile(r"\d+")


def _strip_prefix(video_id: str) -> str:
    """Remove the v2 `<dataset>_<method>_` or v1 `vid_<n>_` prefix -> identity core."""
    if _V2_PREFIX.match(video_id):
        return _V2_PREFIX.sub("", video_id, count=1)
    return _VID_PREFIX.sub("", video_id)


def extract_identities(video_id: str, source: str) -> set:
    """Return the set of identity tokens involved in a video."""
    core = _strip_prefix(video_id)
    if source == "CelebDF":
        ids = _CELEB_ID.findall(core)
        if ids:
            return {f"celeb_{x}" for x in ids}
        # YouTube-real: no idNN, leading number is the clip identity
        m = _LEADING_NUM.search(core)
        token = m.group(0) if m else core
        return {f"yt_{token}"}
    elif source == "FFpp":
        # core: '000' (real) | '000_003' (swap pair) | '01_02__scene__HASH' (actor pair)
        head = core.split("__", 1)[0]
        nums = _LEADING_NUM.findall(head)
        if nums:
            return {f"ff_{n}" for n in nums}
        return {f"ff_{head}"}
    elif source == "DFD":
        head = core.split("__", 1)[0]          # '24' or '12_13'
        nums = _LEADING_NUM.findall(head)
        if nums:
            return {f"dfd_{n}" for n in nums}
        return {f"dfd_{head}"}
    elif source == "DFDC":
        # public DFDC frame dump exposes no actor identity -> group by video hash
        # (each video is its own component; split is video-level, not identity-level).
        return {f"dfdc_{core}"}
    # unknown source -> fall back to the whole core as one identity
    return {f"other_{core}"}


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:       # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(records):
    """records: iterable of (video_id, source). Returns dict video_id->group_id.

    group_id is the connected-component root (a stable identity token string).
    """
    uf = UnionFind()
    vid_idents = {}
    for video_id, source in records:
        idents = sorted(extract_identities(video_id, source))
        vid_idents[video_id] = idents
        first = idents[0]
        uf.find(first)
        for other in idents[1:]:
            uf.union(first, other)
    return {vid: uf.find(idents[0]) for vid, idents in vid_idents.items()}, vid_idents
