"""Phase 4 pairwise descriptor matching (Milestone 2 Step 2).

Returns tentative matches only (knnMatch + Lowe's ratio test).
No geometric filtering (RANSAC / fundamental matrix) is done here — those
are responsibilities of the next phase.

Data structures are shaped so that handoff to geometric estimation is a
one-liner: `result.points_a(fs_a), result.points_b(fs_b)` return the
`(N, 2)` float32 arrays that `cv2.findFundamentalMat` consumes directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .features import FeatureSet, apply_grid_filter
from .frame_loader import Frame


SUPPORTED_MATCHERS = ("flann", "bf")
SUPPORTED_PIPELINES = ("sift", "superpoint")
DEFAULT_RATIO = 0.75


@dataclass
class FrameMatchResult:
    """Tentative descriptor matches between two frames.

    When the spatial grid filter is active, `fs_a_filtered` / `fs_b_filtered`
    hold the sub-sampled FeatureSets that were actually fed to the matcher.
    Match indices (queryIdx / trainIdx) are relative to those filtered sets,
    not to the original full FeatureSets.

    `points_a()` / `points_b()` always reference the filtered sets automatically,
    so callers do not need to distinguish between the two cases — the handoff to
    `cv2.findFundamentalMat` is always a one-liner.
    """
    idx_a: int
    idx_b: int
    name_a: str
    name_b: str
    num_desc_a: int          # keypoints in the full (unfiltered) FeatureSet A
    num_desc_b: int          # keypoints in the full (unfiltered) FeatureSet B
    num_raw_matches: int
    ratio_threshold: float
    matcher: str
    mutual: bool
    grid_filtered: bool = False
    tentative_matches: List[cv2.DMatch] = field(default_factory=list)
    # Filtered FeatureSets used for matching (same as originals when grid_filtered=False)
    fs_a_filtered: Optional[FeatureSet] = field(default=None, repr=False)
    fs_b_filtered: Optional[FeatureSet] = field(default=None, repr=False)

    @property
    def num_tentative(self) -> int:
        return len(self.tentative_matches)

    @property
    def ratio_kept(self) -> float:
        if self.num_raw_matches == 0:
            return 0.0
        return self.num_tentative / self.num_raw_matches

    def points_a(self, fs_a: Optional[FeatureSet] = None) -> np.ndarray:
        """Return (N, 2) float32 coordinates of matched keypoints in frame A.

        When the grid filter was active, pass no argument (or pass
        `result.fs_a_filtered`) — the filtered set is stored internally.
        For backward-compatibility, an explicit `fs_a` argument is still
        accepted and used if provided.
        """
        source = fs_a if fs_a is not None else self.fs_a_filtered
        if source is None:
            raise ValueError("No FeatureSet available; pass fs_a explicitly.")
        return np.float32([source.keypoints[m.queryIdx].pt
                           for m in self.tentative_matches]).reshape(-1, 2)

    def points_b(self, fs_b: Optional[FeatureSet] = None) -> np.ndarray:
        """Return (N, 2) float32 coordinates of matched keypoints in frame B."""
        source = fs_b if fs_b is not None else self.fs_b_filtered
        if source is None:
            raise ValueError("No FeatureSet available; pass fs_b explicitly.")
        return np.float32([source.keypoints[m.trainIdx].pt
                           for m in self.tentative_matches]).reshape(-1, 2)


def _make_matcher(method: str, descriptor_type: str = "float"):
    """Instantiate a matcher.

    `descriptor_type`:
      - "float"  → SIFT-like L2 descriptors (default)
      - "binary" → AKAZE/ORB Hamming descriptors
    """
    m = method.lower()
    if m == "flann":
        if descriptor_type == "float":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    if m == "bf":
        norm = cv2.NORM_L2 if descriptor_type == "float" else cv2.NORM_HAMMING
        return cv2.BFMatcher(norm, crossCheck=False)
    raise ValueError(f"Unsupported matcher {method!r}; expected one of {SUPPORTED_MATCHERS}")


def _apply_ratio_test(
    knn_matches: Sequence[Sequence[cv2.DMatch]],
    ratio: float,
) -> List[cv2.DMatch]:
    kept: List[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio * n.distance:
            kept.append(m)
    return kept


def match_pair(
    fs_a: FeatureSet,
    fs_b: FeatureSet,
    idx_a: int = -1,
    idx_b: int = -1,
    method: str = "flann",
    ratio: float = DEFAULT_RATIO,
    mutual: bool = False,
    grid_filter: bool = True,
    grid_rows: int = 4,
    grid_cols: int = 5,
    grid_max_per_cell: int = 15,
    pipeline: str = "sift",
) -> FrameMatchResult:
    """Match descriptors of two frames.

    Two pipelines are supported via the `pipeline` argument (default "sift"):

    * `pipeline="sift"`  : classical SIFT/AKAZE descriptors + FLANN/BF +
        Lowe's ratio test (+ optional mutual-best + spatial grid filter).
        All existing kwargs (`method`, `ratio`, `mutual`, `grid_filter`,
        `grid_*`) apply and behave exactly as before.
    * `pipeline="superpoint"` : LightGlue assignment on SuperPoint
        descriptors. Bypasses FLANN/BF; the `method` / `ratio` / `mutual`
        / `grid_*` kwargs are ignored (LightGlue has its own learned
        assignment). Delegates to `src.deep_matching.match_pair_deep`.

    When `grid_filter=True` (default, SIFT path only), each FeatureSet is
    spatially sub-sampled with `apply_grid_filter()` before matching so
    every grid cell contributes equally. Original FeatureSets are never
    mutated; match indices are relative to the filtered copies stored in
    `FrameMatchResult.fs_a_filtered` / `.fs_b_filtered`.
    """
    if pipeline.lower() == "superpoint":
        # Lazy import keeps torch / lightglue optional.
        from .deep_matching import match_pair_deep
        return match_pair_deep(fs_a, fs_b, idx_a=idx_a, idx_b=idx_b)

    if pipeline.lower() != "sift":
        raise ValueError(
            f"Unsupported pipeline {pipeline!r}; expected one of {SUPPORTED_PIPELINES}."
        )

    if fs_a.descriptors is None or fs_b.descriptors is None:
        raise ValueError("Both FeatureSets must have descriptors.")
    if fs_a.num_keypoints < 2 or fs_b.num_keypoints < 2:
        raise ValueError("Need at least 2 keypoints per frame for knnMatch(k=2).")

    if grid_filter:
        fa = apply_grid_filter(fs_a, grid_rows=grid_rows, grid_cols=grid_cols,
                               max_per_cell=grid_max_per_cell)
        fb = apply_grid_filter(fs_b, grid_rows=grid_rows, grid_cols=grid_cols,
                               max_per_cell=grid_max_per_cell)
    else:
        fa, fb = fs_a, fs_b

    if fa.num_keypoints < 2 or fb.num_keypoints < 2:
        return FrameMatchResult(
            idx_a=idx_a, idx_b=idx_b,
            name_a=fs_a.frame_name, name_b=fs_b.frame_name,
            num_desc_a=fs_a.num_keypoints, num_desc_b=fs_b.num_keypoints,
            num_raw_matches=0, ratio_threshold=ratio,
            matcher=method.lower(), mutual=mutual,
            grid_filtered=grid_filter,
            tentative_matches=[],
            fs_a_filtered=fa, fs_b_filtered=fb,
        )

    descriptor_type = "float" if fa.descriptors.dtype != np.uint8 else "binary"
    matcher = _make_matcher(method, descriptor_type=descriptor_type)

    knn_ab = matcher.knnMatch(fa.descriptors, fb.descriptors, k=2)
    tentative = _apply_ratio_test(knn_ab, ratio)

    if mutual and tentative:
        knn_ba = matcher.knnMatch(fb.descriptors, fa.descriptors, k=2)
        reverse = _apply_ratio_test(knn_ba, ratio)
        reverse_best: Dict[int, int] = {m.queryIdx: m.trainIdx for m in reverse}
        tentative = [
            m for m in tentative
            if reverse_best.get(m.trainIdx, -1) == m.queryIdx
        ]

    return FrameMatchResult(
        idx_a=idx_a, idx_b=idx_b,
        name_a=fs_a.frame_name, name_b=fs_b.frame_name,
        num_desc_a=fs_a.num_keypoints, num_desc_b=fs_b.num_keypoints,
        num_raw_matches=len(knn_ab),
        ratio_threshold=ratio,
        matcher=method.lower(),
        mutual=mutual,
        grid_filtered=grid_filter,
        tentative_matches=tentative,
        fs_a_filtered=fa,
        fs_b_filtered=fb,
    )


def match_frame_pairs(
    feature_sets: Sequence[FeatureSet],
    pairs: Sequence[Tuple[int, int]],
    method: str = "flann",
    ratio: float = DEFAULT_RATIO,
    mutual: bool = False,
    grid_filter: bool = True,
    grid_rows: int = 4,
    grid_cols: int = 5,
    grid_max_per_cell: int = 15,
    progress: bool = False,
    pipeline: str = "sift",
) -> List[FrameMatchResult]:
    """Run `match_pair` over a list of (i, j) index pairs.

    Pass `pipeline="superpoint"` to route all pairs through LightGlue
    (see `match_pair` for the full behavioral contract of each pipeline).
    """
    results: List[FrameMatchResult] = []
    for k, (i, j) in enumerate(pairs):
        if i == j:
            continue
        res = match_pair(
            feature_sets[i], feature_sets[j],
            idx_a=i, idx_b=j,
            method=method, ratio=ratio, mutual=mutual,
            grid_filter=grid_filter,
            grid_rows=grid_rows, grid_cols=grid_cols,
            grid_max_per_cell=grid_max_per_cell,
            pipeline=pipeline,
        )
        results.append(res)
        if progress:
            if pipeline.lower() == "superpoint":
                print(f"  [{k + 1:3d}/{len(pairs)}] ({i},{j})  "
                      f"lightglue_matches={res.num_tentative:4d}")
            else:
                print(f"  [{k + 1:3d}/{len(pairs)}] ({i},{j})  "
                      f"raw={res.num_raw_matches:5d}  "
                      f"kept={res.num_tentative:4d}  "
                      f"({100 * res.ratio_kept:5.1f}%)")
    return results


def _session_key(frame_name: str) -> str:
    """Group frames by filename prefix (the timestamp before the sub-id)."""
    parts = frame_name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return frame_name


def select_pairs(
    frames: Sequence[Frame],
    policy: str = "session",
    window_size: int = 3,
) -> List[Tuple[int, int]]:
    """Choose which (i, j) index pairs to match.

    Policies:
      - "all"     : every C(N, 2) pair.
      - "session" : all pairs within each same-session group (default).
      - "window"  : sliding temporal window; match frame i with i+1..i+K
                    within the same session only.
    """
    p = policy.lower()
    n = len(frames)

    if p == "all":
        return list(combinations(range(n), 2))

    groups: Dict[str, List[int]] = {}
    for i, f in enumerate(frames):
        groups.setdefault(_session_key(f.name), []).append(i)

    pairs: List[Tuple[int, int]] = []
    if p == "session":
        for indices in groups.values():
            pairs.extend(combinations(indices, 2))
        return sorted(pairs)

    if p == "window":
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        for indices in groups.values():
            for pos, i in enumerate(indices):
                for j in indices[pos + 1: pos + 1 + window_size]:
                    pairs.append((i, j))
        return sorted(pairs)

    raise ValueError(f"Unknown policy {policy!r}; expected all / session / window.")


def parse_pairs_arg(arg: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """Parse a CLI string like '0,1 3,7 4,10' into [(0,1),(3,7),(4,10)]."""
    if not arg:
        return None
    out: List[Tuple[int, int]] = []
    for token in arg.split():
        try:
            a, b = token.split(",")
            out.append((int(a), int(b)))
        except ValueError as e:
            raise ValueError(f"Bad pair token {token!r}, expected 'i,j'.") from e
    return out


def draw_tentative_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    result: FrameMatchResult,
    fs_a: Optional[FeatureSet] = None,
    fs_b: Optional[FeatureSet] = None,
    max_draw: int = 80,
) -> np.ndarray:
    """Return a side-by-side image of the two frames with matches drawn.

    Uses the filtered FeatureSets stored in `result` by default.
    Pass explicit `fs_a` / `fs_b` to override (e.g. to draw on the full sets).
    """
    kps_a = (fs_a if fs_a is not None else result.fs_a_filtered).keypoints
    kps_b = (fs_b if fs_b is not None else result.fs_b_filtered).keypoints
    matches = result.tentative_matches
    if len(matches) > max_draw:
        indices = np.linspace(0, len(matches) - 1, max_draw, dtype=int)
        matches = [matches[i] for i in indices]
    return cv2.drawMatches(
        img_a, kps_a,
        img_b, kps_b,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
