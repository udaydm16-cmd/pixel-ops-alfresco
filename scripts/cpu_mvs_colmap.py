import os
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path

IMAGE_DIR = Path("project/dense/images")
OUTPUT_DIR = Path("project/cpu_mvs_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the existing COLMAP sparse reconstruction as the geometric reference.
SPARSE_PLY = "project/sparse_sparse.ply"

# Conservative CPU settings.
IMAGE_WIDTH = 1200
MAX_PAIRS = 80
MIN_DISPARITY = 0
NUM_DISPARITIES = 256

def resize_image(img):
    h, w = img.shape[:2]
    scale = IMAGE_WIDTH / w
    return cv2.resize(img, (IMAGE_WIDTH, int(h * scale)))

def main():
    images = sorted(IMAGE_DIR.glob("*.jpg"))

    if not images:
        raise RuntimeError("No images found.")

    print("Images:", len(images))
    print("CPU multi-view reconstruction starting...")

    # We will build depth only from neighbouring calibrated views.
    pairs = []
    for i in range(len(images) - 1):
        pairs.append((images[i], images[i + 1]))

    if len(pairs) > MAX_PAIRS:
        idx = np.linspace(0, len(pairs) - 1, MAX_PAIRS).astype(int)
        pairs = [pairs[i] for i in idx]

    print("Stereo pairs:", len(pairs))

    all_points = []
    all_colors = []

    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=7,
        P1=8 * 3 * 7 * 7,
        P2=32 * 3 * 7 * 7,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2
    )

    for n, (left_path, right_path) in enumerate(pairs, 1):
        print(f"[{n}/{len(pairs)}] {left_path.name} -> {right_path.name}")

        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))

        if left is None or right is None:
            continue

        left = resize_image(left)
        right = resize_image(right)

        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        valid = (
            np.isfinite(disparity) &
            (disparity > 2) &
            (disparity < NUM_DISPARITIES)
        )

        if valid.sum() < 100:
            continue

        # Approximate perspective reconstruction.
        h, w = gray_l.shape
        focal = 0.9 * w
        baseline = 0.05

        yy, xx = np.indices((h, w))

        z = (focal * baseline) / disparity
        x = (xx - w / 2) * z / focal
        y = (yy - h / 2) * z / focal

        pts = np.stack((x, y, z), axis=-1)[valid]
        colors = left[valid][:, ::-1] / 255.0

        # Remove extreme stereo failures.
        good = (
            np.isfinite(pts).all(axis=1) &
            (pts[:, 2] > 0.1) &
            (pts[:, 2] < 20.0)
        )

        all_points.append(pts[good])
        all_colors.append(colors[good])

    if not all_points:
        raise RuntimeError("No valid stereo points generated.")

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)

    print("Raw CPU points:", len(points))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Remove isolated stereo noise.
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=1.5
    )

    # Keep a manageable final cloud.
    pcd = pcd.voxel_down_sample(0.01)

    output = OUTPUT_DIR / "dense_pointcloud.ply"
    o3d.io.write_point_cloud(str(output), pcd, write_ascii=False)

    print("Final CPU points:", len(pcd.points))
    print("Saved:", output)

if __name__ == "__main__":
    main()
