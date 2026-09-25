"""
Pixel-Ops - Lighthouse dataset
Loads the fused dense point cloud and saves an offscreen screenshot,
for submission as the required "rendered/screenshot view" deliverable.

Usage:
    python3 render_pointcloud.py <path/to/fused.ply> [output.png]
"""

import sys
import open3d as o3d


def render(ply_path: str, output_png: str = "reconstruction_screenshot.png") -> None:
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Loaded {len(pcd.points)} points from {ply_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1600, height=1200)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = [0.05, 0.05, 0.05]

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_png, do_render=True)
    vis.destroy_window()
    print(f"Saved screenshot to {output_png}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 render_pointcloud.py <fused.ply> [output.png]")
    ply_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "reconstruction_screenshot.png"
    render(ply_path, out)
