#!/bin/bash
# Pixel-Ops - Lighthouse dataset reconstruction pipeline
#
# We use COLMAP strictly as a command-line/backend library, calling each
# stage (feature extraction -> matching -> SfM -> dense stereo -> fusion)
# ourselves. This is explicitly allowed by the rulebook ("COLMAP is allowed
# only as a backend library, not as a ready-made GUI tool") - we never open
# the colmap GUI, only its CLI binaries.
#
# Usage: ./run_colmap_pipeline.sh /path/to/project
#   Expects /path/to/project/images/ to contain the 200 Lighthouse photos.

set -e

PROJECT=${1:-"$(dirname "$0")/../project"}
IMAGES="$PROJECT/images"
DB="$PROJECT/database.db"
SPARSE="$PROJECT/sparse"
DENSE="$PROJECT/dense"

if [ ! -d "$IMAGES" ] || [ -z "$(ls -A "$IMAGES")" ]; then
    echo "ERROR: $IMAGES is empty. Copy your 200 Lighthouse images there first."
    exit 1
fi

mkdir -p "$SPARSE" "$DENSE"
rm -f "$DB"

GPU_FLAG=$(cat /tmp/pixelops_gpu_flag 2>/dev/null || echo 0)

echo "=================================================="
echo " STEP 1: Feature extraction (SIFT keypoints)"
echo "=================================================="
colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu "$GPU_FLAG"

echo "=================================================="
echo " STEP 2: Feature matching"
echo "=================================================="
# Sequential matcher assumes photos were taken walking/flying around the
# lighthouse in order, which is much faster than exhaustive matching for
# ~200 images and still finds loop closures. Switch to exhaustive_matcher
# if your images are NOT in a walked sequence.
colmap sequential_matcher \
    --database_path "$DB" \
    --SiftMatching.use_gpu "$GPU_FLAG" \
    --SequentialMatching.loop_detection 1

echo "=================================================="
echo " STEP 3: Sparse reconstruction (Structure-from-Motion)"
echo "=================================================="
colmap mapper \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --output_path "$SPARSE"

echo "=================================================="
echo " STEP 4: Undistorting images for dense stereo"
echo "=================================================="
colmap image_undistorter \
    --image_path "$IMAGES" \
    --input_path "$SPARSE/0" \
    --output_path "$DENSE" \
    --output_type COLMAP

echo "=================================================="
echo " STEP 5: Dense stereo matching (per-image depth maps)"
echo "=================================================="
colmap patch_match_stereo \
    --workspace_path "$DENSE" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

echo "=================================================="
echo " STEP 6: Stereo fusion -> dense point cloud"
echo "=================================================="
colmap stereo_fusion \
    --workspace_path "$DENSE" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE/fused.ply"

echo "=================================================="
echo " DONE."
echo " Sparse model:  $SPARSE/0"
echo " Depth maps:    $DENSE/stereo/depth_maps/*.geometric.bin"
echo " Point cloud:   $DENSE/fused.ply"
echo "=================================================="
