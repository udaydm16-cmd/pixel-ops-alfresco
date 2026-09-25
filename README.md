# Pixel-Ops — Lighthouse Trial Round

End-to-end pipeline: 200 survey images → dense 3D point cloud → interactive
web viewer hosted on GitHub Pages.

## 0. Rule compliance (read this first)
- COLMAP is used **only via its command-line binaries**, called from our own
  shell/Python scripts — we never open a GUI reconstruction app. This matches
  rule 6.3: *"COLMAP is allowed only as a backend library, not as a ready-made
  GUI tool."*
- No pre-built point clouds or external 3D models are used — everything is
  derived only from the issued Lighthouse images.
- Be ready to explain every step below during shortlist review (rule 6.4 / 7).

## 1. Setup
```bash
chmod +x setup.sh scripts/run_colmap_pipeline.sh
./setup.sh
```

## 2. Add your images
Copy all 200 Lighthouse images into:
```
project/images/
```

## 3. Run the reconstruction pipeline
```bash
./scripts/run_colmap_pipeline.sh project
```
This runs, in order: feature extraction → matching → SfM → undistortion →
dense stereo → fusion. Outputs:
- `project/dense/fused.ply` — the dense point cloud (submission deliverable)
- `project/dense/stereo/depth_maps/*.geometric.bin` — raw depth maps

⏱ For ~200 images on CPU this can take a few hours — this is why the
rulebook says "work overnight." If you have an NVIDIA GPU, `setup.sh`
auto-detects it and speeds up feature extraction + matching significantly.

## 4. Export depth map images (submission deliverable)
```bash
python3 scripts/export_depth_maps.py project --limit 20
```
Produces colorized PNGs in `project/dense/depth_maps_png/`.

## 5. Render a screenshot (submission deliverable)
```bash
python3 scripts/render_pointcloud.py project/dense/fused.ply reconstruction_screenshot.png
```

## 6. Prepare the point cloud for the web viewer
```bash
python3 scripts/prepare_web_model.py project/dense/fused.ply web/model.ply 0.02
```
This downsamples + cleans the cloud and writes `web/model.ply`, which the
viewer in `web/index.html` loads directly. If the printed file size is close
to 100MB, re-run with a larger voxel size (e.g. `0.03`) to shrink it further.

## 7. Put it on GitHub Pages
```bash
cd web
git init
git add index.html model.ply
git commit -m "Lighthouse 3D viewer"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
Then on GitHub: **Settings → Pages → Source: `main` branch, `/ (root)` folder → Save.**
GitHub gives you a link like:
```
https://<your-username>.github.io/<your-repo>/
```
Open it, click **"OK — View 3D Model,"** and the point cloud loads with full
mouse control (left-drag to orbit in any direction, right-drag to pan, scroll
to zoom).

> If `model.ply` is over 100MB, GitHub will reject the push — either increase
> the voxel size in step 6, or use [Git LFS](https://git-lfs.com/).

## 8. Fill in the documentation
Use `docs/documentation_template.md` (max 1 page) — fill in your team's
actual approach notes and known limitations before printing/submitting.

## Folder structure
```
pixel-ops-lighthouse/
├── setup.sh
├── scripts/
│   ├── run_colmap_pipeline.sh
│   ├── export_depth_maps.py
│   ├── render_pointcloud.py
│   └── prepare_web_model.py
├── project/
│   └── images/            <- put your 200 Lighthouse photos here
├── docs/
│   └── documentation_template.md
└── web/
    ├── index.html          <- the GitHub Pages viewer
    └── model.ply           <- generated in step 6
```
