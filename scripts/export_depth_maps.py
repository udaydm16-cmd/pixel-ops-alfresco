"""
Pixel-Ops - Lighthouse dataset
Reads COLMAP's raw binary depth-map format and exports each one as a
colorized PNG, for submission as the required "depth map(s)" deliverable.

COLMAP depth map (.bin) format:
  - ASCII header "WIDTH&HEIGHT&CHANNELS&"
  - followed by WIDTH*HEIGHT*CHANNELS little-endian float32 values,
    row-major, channel-interleaved.

Usage:
    python3 export_depth_maps.py <project_dir> [--max-depth 50] [--limit 20]
"""

import argparse
import os
import struct

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def read_colmap_array(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = b""
        amp_count = 0
        while amp_count < 3:
            byte = f.read(1)
            if not byte:
                raise ValueError(f"Malformed COLMAP array header in {path}")
            header += byte
            if byte == b"&":
                amp_count += 1
        width, height, channels = (int(x) for x in header.decode("ascii").rstrip("&").split("&"))

        data = np.frombuffer(f.read(), dtype=np.float32)
        expected = width * height * channels
        if data.size != expected:
            raise ValueError(
                f"Unexpected array size in {path}: got {data.size}, expected {expected}"
            )
        data = data.reshape((height, width, channels)) if channels > 1 else data.reshape((height, width))
    return data


def export_depth_maps(project_dir: str, max_depth: float, limit: int) -> None:
    depth_dir = os.path.join(project_dir, "dense", "stereo", "depth_maps")
    out_dir = os.path.join(project_dir, "dense", "depth_maps_png")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(depth_dir):
        raise SystemExit(f"Depth map folder not found: {depth_dir}\n"
                          f"Run scripts/run_colmap_pipeline.sh first.")

    bin_files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".geometric.bin"))
    if limit:
        bin_files = bin_files[:limit]

    if not bin_files:
        raise SystemExit(f"No .geometric.bin depth maps found in {depth_dir}")

    for i, fname in enumerate(bin_files):
        depth = read_colmap_array(os.path.join(depth_dir, fname))

        # Clip and normalize for visualization. Zero = invalid/unestimated depth.
        valid = depth > 0
        clipped = np.clip(depth, 0, max_depth)
        norm = np.zeros_like(clipped)
        if valid.any():
            norm[valid] = clipped[valid] / max_depth

        colored = cm.viridis(norm)
        colored[~valid] = [0, 0, 0, 1]  # mark invalid pixels black

        out_name = os.path.splitext(os.path.splitext(fname)[0])[0] + "_depth.png"
        out_path = os.path.join(out_dir, out_name)
        plt.imsave(out_path, colored)
        print(f"[{i + 1}/{len(bin_files)}] saved {out_path}")

    print(f"\nDone. {len(bin_files)} depth map PNGs written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "project"))
    parser.add_argument("--max-depth", type=float, default=50.0,
                         help="Depth value (in scene units) used as the top of the color scale.")
    parser.add_argument("--limit", type=int, default=20,
                         help="Only export this many depth maps (0 = all). Default 20 to keep submissions light.")
    args = parser.parse_args()
    export_depth_maps(args.project_dir, args.max_depth, args.limit)
