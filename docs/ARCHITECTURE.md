# System Architecture

## Overview

The system is designed as a modular pipeline that combines geometric reasoning and visual matching to perform accurate pixel re-projection across multiple drone frames.

Due to strong parallax and non-planar scene structure, the architecture avoids global planar assumptions and instead relies on **epipolar geometry and local refinement techniques**.

---

## High-Level Pipeline

The system operates in two main stages:

### 1. Offline Preprocessing

* Frame preparation
* Feature extraction
* Pairwise geometric estimation

### 2. Online Query Execution

* Pixel selection
* Correspondence estimation
* Refinement and validation

---

## Architecture Components

### 1. Frame Loader

Responsible for:

* Loading input frames
* Organizing frame metadata
* Preparing data structures for processing

---

### 2. Telemetry Extraction Module

Responsible for:
- Detecting fixed overlay regions in each frame
- Extracting on-screen telemetry using OCR
- Parsing structured metadata fields

Typical extracted metadata:
- Latitude / Longitude
- Heading / yaw
- Altitude
- Speed
- Timestamp
- Drone label or flight state

Purpose:
- Provide coarse inter-frame spatial and temporal information
- Enable filtering and ranking of frame pairs
- Support consistency checks during matching

Note:
Telemetry is used only as a coarse prior and supporting signal.
It is not used as the primary reprojection mechanism, as pixel-level accuracy is achieved through vision-based geometric alignment.

### 3. Image Preprocessing

Key operations:

* Removal of overlay (HUD, telemetry text)
* Cropping irrelevant regions
* Contrast normalization (e.g., CLAHE)
* Optional denoising

Purpose:

* Improve robustness of feature detection and matching

---

### 4. Feature Extraction

Extracts keypoints and descriptors from each frame.

Recommended implementation:

* **SuperPoint + LightGlue** (preferred for robustness)
* Alternative: SIFT or AKAZE

Output:

* Keypoints
* Descriptors per frame

---

### 5. Pairwise Matching Engine

For each pair of frames:

* Feature matching
* Outlier rejection (RANSAC)
* Inlier selection

---

### 6. Geometric Estimation

For each frame pair, estimate:

* **Fundamental Matrix (F)**

This defines the epipolar constraint between the two views.

Output:

* Fundamental matrix
* Inlier correspondences
* Matching quality metrics

---

### 7. Epipolar Transfer Engine (Core Module)

This is the main component responsible for point transfer.

#### Input:

* Source pixel (u, v)
* Source frame
* Target frame
* Fundamental matrix

#### Process:

1. Compute epipolar line in target image
2. Sample candidate points along the line
3. Evaluate similarity using patch matching
4. Select best candidate

---

### 8. Local Patch Matching

Used to resolve ambiguity along the epipolar line.

Methods:

* Normalized Cross-Correlation (NCC)
* Descriptor similarity

Output:

* Best matching candidate
* Similarity score

---

### 9. Local Refinement Module

Improves precision of predicted point.

Techniques:

* Local window search
* Sub-pixel refinement
* Local affine transformation estimation

---

### 10. Multi-Frame Consistency Module

Enhances robustness by combining predictions across frames.

Strategies:

* Cross-frame validation
* Multi-view agreement
* Optional triangulation for 3D consistency

---

### 11. Confidence Estimation

Each prediction includes a confidence score based on:

* Matching quality
* Patch similarity
* Geometric consistency
* Multi-frame agreement

---

### 12. Evaluation Module

Responsible for performance analysis.

Metrics:

* Mean pixel error
* Median error
* Success rate (≤ 10 pixels)

Also generates:

* Visual overlays
* Comparison plots

---

## Data Flow

### Offline Stage

1. Load frames
2. Extract telemetry from overlays using OCR
3. Preprocess images
4. Extract features
5. Compute pairwise matches
6. Estimate fundamental matrices
7. Store results

---

### Online Stage

1. User selects pixel in source frame
2. Rank candidate target frames using telemetry proximity (optional)
3. For each target frame:
   - Compute epipolar line
   - Perform guided search
   - Refine prediction
   - Compute confidence
4. Aggregate results

---

## Design Decisions

### Why use OCR-based Telemetry?

- Provides coarse spatial and temporal metadata
- Helps filter and prioritize frame pairs
- Improves robustness through consistency checks
- Complements vision-based methods

Note:
Telemetry is not used directly for reprojection due to limited accuracy and lack of depth information.

### Why not Homography?

* Scene is not planar
* Strong parallax invalidates global warp assumptions

### Why Epipolar Geometry?

* Provides physically correct constraint
* Reduces search space from 2D to 1D

### Why Local Matching?

* Resolves ambiguity along epipolar line
* Handles appearance variations

### Why Multi-Frame Consistency?

* Improves robustness
* Reduces outliers

---

## Extensibility

The architecture supports future extensions such as:

* 3D reconstruction (SfM)
* Depth estimation integration
* Object-aware tracking (e.g., vehicle detection)
* Learned correspondence refinement

---

## Summary

This architecture combines:

* Geometric constraints (epipolar geometry)
* Visual similarity (patch matching)
* Local refinement
* Multi-view consistency

to achieve accurate and robust pixel re-projection under challenging real-world conditions.
