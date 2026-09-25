"""
Pixel-Ops - Lighthouse dataset
Downsamples the dense fused.ply so it's light enough to load fast in a
browser (GitHub Pages / mobile), and writes it as binary PLY, which
Three.js's PLYLoader reads directly.

Usage:
    python3 prepare_web_model.py <path/to/fused.ply> [web/model.ply] [voxel_size]
"""

import os
import sys
import open3d as o3d


def prepare(input_ply: str, output_ply: str, voxel_size: float) -> None:
    pcd = o3d.io.read_point_cloud(input_ply)
    n_before = len(pcd.points)
    print(f"Original points: {n_before}")

    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    n_after = len(pcd_down.points)
    print(f"Downsampled points: {n_after} (voxel_size={voxel_size})")

    # Remove obvious outliers left over from noisy stereo matches.
    pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After outlier removal: {len(pcd_clean.points)}")

    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    o3d.io.write_point_cloud(output_ply, pcd_clean, write_ascii=False)

    size_mb = os.path.getsize(output_ply) / (1024 * 1024)
    print(f"Saved web-ready point cloud to {output_ply} ({size_mb:.1f} MB)")
    if size_mb > 90:
        print("WARNING: file is close to GitHub's 100MB limit. "
              "Increase voxel_size to shrink it further, or use Git LFS.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 prepare_web_model.py <fused.ply> [web/model.ply] [voxel_size]")
    input_ply = sys.argv[1]
    output_ply = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "..", "web", "model.ply")
    voxel_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    prepare(input_ply, output_ply, voxel_size)
