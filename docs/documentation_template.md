# Pixel-Ops — Trial Round Documentation
**Team Name:** _______________
**Dataset:** Lighthouse

## 1. Approach / Pipeline
We reconstructed the Lighthouse scene using a classical Structure-from-Motion
(SfM) + Multi-View Stereo (MVS) pipeline, orchestrated entirely by our own
scripts calling COLMAP's command-line binaries as a backend library (no GUI
reconstruction tool was used):

1. **Feature extraction** — SIFT keypoints extracted per image (`colmap feature_extractor`).
2. **Feature matching** — Sequential matching across the ordered survey images, with loop-closure detection (`colmap sequential_matcher`).
3. **Sparse reconstruction** — Incremental SfM: triangulation + bundle adjustment to recover camera poses and a sparse point cloud (`colmap mapper`).
4. **Image undistortion** — Images rectified for dense matching (`colmap image_undistorter`).
5. **Dense stereo** — Per-image depth maps computed via PatchMatch stereo with geometric consistency (`colmap patch_match_stereo`).
6. **Stereo fusion** — All depth maps fused into a single dense colored point cloud (`colmap stereo_fusion`).
7. **Post-processing** — Statistical outlier removal and voxel downsampling (Open3D) for a clean, web-viewable model.

## 2. Libraries / Tools Used
- **COLMAP** (CLI only) — SfM + MVS backend
- **Open3D** — point cloud I/O, outlier removal, downsampling, offscreen rendering
- **OpenCV / NumPy** — image and array handling in custom scripts
- **Three.js** — browser-based point cloud viewer (GitHub Pages)

## 3. Known Limitations / Incomplete Sections
- _(e.g., "Base of the lighthouse is under-reconstructed due to fewer overlapping viewpoints at ground level.")_
- _(e.g., "Some highly reflective/uniform-texture surfaces produced sparse or noisy depth estimates.")_
- _(e.g., "Point cloud is not meshed/textured — only a colored point cloud was produced, per the required deliverable.")_

## 4. Team Explainability Notes
Every member should be able to explain, if asked:
- Why sequential (vs exhaustive) matching was chosen
- What bundle adjustment does during `colmap mapper`
- What a depth map's `geometric.bin` values represent, and how it was parsed
- Why voxel downsampling + outlier removal were applied before the web export
