import os
import struct
import cv2
import numpy as np
import open3d as o3d

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
IMAGE_DIR = "project/dense/images"
MODEL_DIR = "project/dense/sparse"
OUTPUT_DIR = "project/cpu_dense"

TARGET_WIDTH = 1600
PAIR_STEP = 8
MAX_PAIRS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "depth_maps"), exist_ok=True)


# ------------------------------------------------------------
# COLMAP binary readers
# ------------------------------------------------------------
def read_cameras(path):
    cameras = {}

    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num):
            camera_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            # PINHOLE = 4 parameters: fx, fy, cx, cy
            params = struct.unpack("<dddd", f.read(32))

            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }

    return cameras


def qvec_to_rotmat(q):
    qw, qx, qy, qz = q

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,
         2*qx*qy - 2*qz*qw,
         2*qx*qz + 2*qy*qw],

        [2*qx*qy + 2*qz*qw,
         1 - 2*qx*qx - 2*qz*qz,
         2*qy*qz - 2*qx*qw],

        [2*qx*qz - 2*qy*qw,
         2*qy*qz + 2*qx*qw,
         1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)


def read_images(path):
    images = []

    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num):
            image_id = struct.unpack("<i", f.read(4))[0]

            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))

            camera_id = struct.unpack("<i", f.read(4))[0]

            name_bytes = bytearray()

            while True:
                b = f.read(1)
                if b == b"\x00":
                    break
                name_bytes.extend(b)

            name = name_bytes.decode("utf-8")

            num_points = struct.unpack("<Q", f.read(8))[0]

            # Skip 2D observations.
            f.seek(num_points * 24, 1)

            images.append({
                "id": image_id,
                "name": name,
                "camera_id": camera_id,
                "R": qvec_to_rotmat(qvec),
                "t": np.array(tvec, dtype=np.float64),
            })

    return images


# ------------------------------------------------------------
# Load COLMAP reconstruction
# ------------------------------------------------------------
cameras = read_cameras(os.path.join(MODEL_DIR, "cameras.bin"))
images = read_images(os.path.join(MODEL_DIR, "images.bin"))

images.sort(key=lambda x: x["name"])

camera = cameras[images[0]["camera_id"]]

fx, fy, cx, cy = camera["params"]
orig_w = camera["width"]
orig_h = camera["height"]

scale = TARGET_WIDTH / orig_w
target_w = TARGET_WIDTH
target_h = int(round(orig_h * scale))

K = np.array([
    [fx * scale, 0, cx * scale],
    [0, fy * scale, cy * scale],
    [0, 0, 1]
], dtype=np.float64)

print("Images:", len(images))
print("Original resolution:", orig_w, "x", orig_h)
print("Processing resolution:", target_w, "x", target_h)
print("Camera matrix:")
print(K)


# ------------------------------------------------------------
# Stereo matcher
# ------------------------------------------------------------
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=256,
    blockSize=5,
    P1=8 * 3 * 5 * 5,
    P2=32 * 3 * 5 * 5,
    disp12MaxDiff=1,
    uniquenessRatio=8,
    speckleWindowSize=80,
    speckleRange=2,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)


all_points = []
all_colors = []


# ------------------------------------------------------------
# Process image pairs
# ------------------------------------------------------------
pairs_done = 0

for i in range(0, len(images) - PAIR_STEP, PAIR_STEP):

    if pairs_done >= MAX_PAIRS:
        break

    a = images[i]
    b = images[i + PAIR_STEP]

    path_a = os.path.join(IMAGE_DIR, a["name"])
    path_b = os.path.join(IMAGE_DIR, b["name"])

    print()
    print("=" * 60)
    print("Pair", pairs_done + 1, "/", MAX_PAIRS)
    print(a["name"])
    print(b["name"])

    img_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
    img_b = cv2.imread(path_b, cv2.IMREAD_COLOR)

    if img_a is None or img_b is None:
        print("Could not read pair. Skipping.")
        continue

    img_a = cv2.resize(
        img_a,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )

    img_b = cv2.resize(
        img_b,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # Relative camera transformation:
    # X_b = R_rel X_a + T_rel
    R_a = a["R"]
    t_a = a["t"]

    R_b = b["R"]
    t_b = b["t"]

    R_rel = R_b @ R_a.T
    T_rel = (t_b - R_rel @ t_a).reshape(3, 1)

    baseline = np.linalg.norm(T_rel)

    print("Baseline:", baseline)

    if baseline < 1e-4:
        print("Baseline too small. Skipping.")
        continue

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K,
        np.zeros(5),
        K,
        np.zeros(5),
        (target_w, target_h),
        R_rel,
        T_rel,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K, np.zeros(5), R1, P1,
        (target_w, target_h),
        cv2.CV_32FC1
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        K, np.zeros(5), R2, P2,
        (target_w, target_h),
        cv2.CV_32FC1
    )

    rect_a = cv2.remap(
        gray_a, map1x, map1y, cv2.INTER_LINEAR
    )

    rect_b = cv2.remap(
        gray_b, map2x, map2y, cv2.INTER_LINEAR
    )

    color_a = cv2.remap(
        img_a, map1x, map1y, cv2.INTER_LINEAR
    )

    print("Computing disparity...")

    disparity = stereo.compute(rect_a, rect_b).astype(np.float32)
    disparity /= 16.0

    valid = (
        np.isfinite(disparity) &
        (disparity > 1.0) &
        (disparity < 256.0)
    )

    valid_count = int(np.count_nonzero(valid))

    print("Valid depth pixels:", valid_count)

    if valid_count < 1000:
        print("Too few valid pixels. Skipping.")
        continue

    # Reconstruct 3D coordinates in the rectified first-camera frame.
    points_rect = cv2.reprojectImageTo3D(disparity, Q)

    points_rect = points_rect[valid]
    colors = color_a[valid][:, ::-1]  # BGR -> RGB

    # Transform rectified camera coordinates back to
    # original first-camera coordinates.
    points_cam = (R1.T @ points_rect.T).T

    # Transform camera coordinates into COLMAP world coordinates.
    points_world = (
        R_a.T @ (points_cam - t_a.reshape(1, 3)).T
    ).T

    # Remove obviously invalid/extreme values.
    finite = np.isfinite(points_world).all(axis=1)

    points_world = points_world[finite]
    colors = colors[finite]

    all_points.append(points_world)
    all_colors.append(colors)

    # Save a visualization of the depth map.
    depth = points_rect[:, 2]

    depth_img = np.zeros((target_h, target_w), dtype=np.float32)
    depth_img[valid] = points_rect[:, 2]

    positive = depth_img > 0

    if np.any(positive):
        dmin = np.percentile(depth[depth > 0], 2)
        dmax = np.percentile(depth[depth > 0], 98)

        depth_norm = np.clip(
            (depth_img - dmin) / (dmax - dmin + 1e-8),
            0,
            1
        )

        depth_png = (depth_norm * 255).astype(np.uint8)

        depth_name = os.path.splitext(a["name"])[0] + "_depth.png"

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                "depth_maps",
                depth_name
            ),
            depth_png
        )

    pairs_done += 1

    print("Points added:", len(points_world))


# ------------------------------------------------------------
# Fuse all reconstructed points
# ------------------------------------------------------------
if not all_points:
    raise RuntimeError("No valid stereo reconstruction was produced.")

points = np.vstack(all_points)
colors = np.vstack(all_colors)

print()
print("=" * 60)
print("Raw reconstructed points:", len(points))

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

print("Downsampling...")

pcd = pcd.voxel_down_sample(voxel_size=0.02)

print("After voxel downsampling:", len(pcd.points))

print("Removing statistical outliers...")

pcd, _ = pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

print("Final points:", len(pcd.points))

output = os.path.join(
    OUTPUT_DIR,
    "dense_pointcloud.ply"
)

o3d.io.write_point_cloud(
    output,
    pcd,
    write_ascii=False
)

print()
print("DONE")
print("Point cloud:", output)
print("Depth maps:", os.path.join(OUTPUT_DIR, "depth_maps"))
