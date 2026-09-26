"""
Pixel-Ops - Lighthouse dataset
Builds a smoothed surface mesh from the sparse SfM point cloud, purely for a
nicer-looking screenshot. This does NOT replace your required point-cloud
deliverable (project/sparse_sparse.ply) - it's an extra, honest visualization
step: Poisson surface reconstruction fits a continuous surface through your
existing real points, it does not invent new geometry.

Usage:
    python3 scripts/make_mesh_screenshot.py project/sparse_sparse.ply mesh_screenshot.png
"""

import sys
import numpy as np
import open3d as o3d


def build_mesh_screenshot(ply_path: str, output_png: str, output_mesh: str = "project/mesh_from_sparse.ply") -> None:
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Loaded {len(pcd.points)} points from {ply_path}")

    # Clean up stray outlier points first so Poisson doesn't chase noise.
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After outlier removal: {len(pcd.points)} points")

    # Poisson needs per-point normals to know which way the surface faces.
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    print("Running Poisson surface reconstruction (this can take 1-3 minutes)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # Poisson fills in low-confidence areas with a blobby "guess" surface where
    # points are sparse or absent. Cut away the lowest-density vertices so the
    # mesh only shows surface actually supported by real points.
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.08)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles after trimming low-density areas")

    o3d.io.write_triangle_mesh(output_mesh, mesh)
    print(f"Saved mesh to {output_mesh}")

    # Offscreen render for the screenshot.
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1600, height=1200)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = [0.05, 0.05, 0.05]
    opt.mesh_show_back_face = True
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_png, do_render=True)
    vis.destroy_window()
    print(f"Saved screenshot to {output_png}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python3 make_mesh_screenshot.py <sparse.ply> <output.png>")
    build_mesh_screenshot(sys.argv[1], sys.argv[2])
