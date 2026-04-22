"""Deep-learning matcher (LightGlue) — new pipeline.

Bypasses FLANN/BF and uses LightGlue on SuperPoint descriptors.
Returns the existing `FrameMatchResult` so RANSAC + epipolar downstream
code keeps working without any change.

Design notes
------------
* LightGlue is learned and already implements its own high-quality
  assignment (no Lowe's ratio test, no mutual-check). The `ratio` /
  `mutual` fields on `FrameMatchResult` are therefore filled with
  sentinel values (0.0 / True) — they are metadata only.
* The spatial grid filter is not applied on the deep pipeline: LightGlue
  relies on its context-aware assignment across *all* features; randomly
  pruning them would strictly hurt accuracy. The filtered-FeatureSets
  fields on the result are still populated (with the full sets) so
  `points_a()` / `points_b()` / `draw_*` helpers keep working unchanged.
* The distance field on the emitted `cv2.DMatch` is filled with
  `(1 - score)` so callers that sort by `.distance` get a sensible order
  (smaller distance = higher confidence).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .deep_features import get_device
from .features import FeatureSet
from .matching import FrameMatchResult


_MATCHER = None


def get_lightglue_matcher(features: str = "superpoint"):
    """Return a cached LightGlue matcher on the resolved device."""
    global _MATCHER
    if _MATCHER is None:
        try:
            from lightglue import LightGlue
        except ImportError as e:
            raise ImportError(
                "LightGlue matching requires the 'lightglue' package.\n"
                "Install with: pip install git+https://github.com/cvg/LightGlue.git"
            ) from e
        device = get_device()
        _MATCHER = LightGlue(features=features).eval().to(device)
    return _MATCHER


def _feature_set_to_lg_input(fs: FeatureSet, device) -> dict:
    """Convert a SuperPoint `FeatureSet` back into a LightGlue input dict.

    Expected by LightGlue for each image:
        {
            "keypoints":   (1, N, 2)  float32
            "descriptors": (1, N, D)  float32
            "image_size":  (1, 2)     float32   # (W, H)
        }
    """
    import torch

    if fs.descriptors is None:
        raise ValueError(
            f"FeatureSet for frame {fs.frame_name!r} has no descriptors."
        )

    kps_np = np.asarray(
        [kp.pt for kp in fs.keypoints], dtype=np.float32
    ).reshape(-1, 2)
    desc_np = np.asarray(fs.descriptors, dtype=np.float32).reshape(
        len(fs.keypoints), -1
    )

    kps_t = torch.from_numpy(kps_np).unsqueeze(0).to(device)   # (1, N, 2)
    desc_t = torch.from_numpy(desc_np).unsqueeze(0).to(device)  # (1, N, D)

    h, w = fs.image_shape
    image_size = torch.tensor([[float(w), float(h)]], device=device)  # (1, 2)

    return {"keypoints": kps_t, "descriptors": desc_t, "image_size": image_size}


def match_pair_deep(
    fs_a: FeatureSet,
    fs_b: FeatureSet,
    idx_a: int = -1,
    idx_b: int = -1,
    features: str = "superpoint",
) -> FrameMatchResult:
    """Match two SuperPoint `FeatureSet`s with LightGlue.

    Returns a `FrameMatchResult` shape-compatible with the SIFT path:
        * `tentative_matches` : list[cv2.DMatch]   (queryIdx -> fs_a, trainIdx -> fs_b)
        * `fs_a_filtered`     : fs_a   (no grid filter on deep path)
        * `fs_b_filtered`     : fs_b
        * `matcher`           : "lightglue"

    `points_a()` / `points_b()` on the result therefore return
    `(N, 2) float32` arrays directly consumable by `cv2.findFundamentalMat`.
    """
    import torch

    if fs_a.num_keypoints == 0 or fs_b.num_keypoints == 0:
        return FrameMatchResult(
            idx_a=idx_a, idx_b=idx_b,
            name_a=fs_a.frame_name, name_b=fs_b.frame_name,
            num_desc_a=fs_a.num_keypoints, num_desc_b=fs_b.num_keypoints,
            num_raw_matches=0, ratio_threshold=0.0,
            matcher="lightglue", mutual=True,
            grid_filtered=False,
            tentative_matches=[],
            fs_a_filtered=fs_a, fs_b_filtered=fs_b,
        )

    device = get_device()
    matcher = get_lightglue_matcher(features=features)

    data = {
        "image0": _feature_set_to_lg_input(fs_a, device),
        "image1": _feature_set_to_lg_input(fs_b, device),
    }

    with torch.no_grad():
        out = matcher(data)

    # LightGlue returns batched outputs; pull the single item.
    # "matches0": (1, N_a) with trainIdx in B or -1 when unmatched.
    # "matching_scores0": (1, N_a) confidence per query.
    matches0 = out["matches0"][0].detach().cpu().numpy()          # (N_a,)
    scores0 = out.get("matching_scores0")
    if scores0 is not None:
        scores0 = scores0[0].detach().cpu().numpy()               # (N_a,)

    dmatches: List[cv2.DMatch] = []
    for qi, ti in enumerate(matches0):
        if ti < 0:
            continue
        score = float(scores0[qi]) if scores0 is not None else 1.0
        # Map confidence -> distance (smaller is better) so sort-by-distance works.
        dist = float(1.0 - score)
        dmatches.append(cv2.DMatch(int(qi), int(ti), 0, dist))

    return FrameMatchResult(
        idx_a=idx_a, idx_b=idx_b,
        name_a=fs_a.frame_name, name_b=fs_b.frame_name,
        num_desc_a=fs_a.num_keypoints, num_desc_b=fs_b.num_keypoints,
        # For LightGlue "raw matches" = number of query features it evaluated.
        # We record the number of kept matches after its internal thresholding.
        num_raw_matches=len(dmatches),
        ratio_threshold=0.0,
        matcher="lightglue",
        mutual=True,                   # LightGlue is mutually-consistent by design
        grid_filtered=False,
        tentative_matches=dmatches,
        fs_a_filtered=fs_a,
        fs_b_filtered=fs_b,
    )


def match_frame_pairs_deep(
    feature_sets: Sequence[FeatureSet],
    pairs: Sequence[Tuple[int, int]],
    features: str = "superpoint",
    progress: bool = False,
) -> List[FrameMatchResult]:
    """Run `match_pair_deep` over a list of (i, j) index pairs."""
    results: List[FrameMatchResult] = []
    for k, (i, j) in enumerate(pairs):
        if i == j:
            continue
        res = match_pair_deep(
            feature_sets[i], feature_sets[j],
            idx_a=i, idx_b=j, features=features,
        )
        results.append(res)
        if progress:
            print(f"  [{k + 1:3d}/{len(pairs)}] ({i},{j})  "
                  f"lightglue_matches={res.num_tentative:4d}")
    return results
