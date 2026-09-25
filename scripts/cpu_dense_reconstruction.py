import os
import struct
import cv2
import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "project/dense/images"
MODEL_DIR = "project/dense/sparse"
SPARSE_PLY = "project/sparse_sparse.ply"

OUTPUT_DIR = "project/cpu_dense_final"
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth_maps")

# Reduced resolution for a 14 GB RAM CPU-only machine.
TARGET_WIDTH = 800

# Adjacent/semi-adjacent images have stronger overlap.
PAIR_STEP = 4

# Keep runtime manageable.
MAX_PAIRS = 20

# Expand sparse reconstruction bounds by 25%.
BOUND_MARGIN = 0.25


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)


# ============================================================
# COLMAP CAMERA READER
# ============================================================

def read_camera(path):

    with open(path, "rb") as f:

        count = struct.unpack("<Q", f.read(8))[0]

        if count != 1:
            raise RuntimeError(
                f"Expected exactly one camera, found {count}"
            )

        camera_id = struct.unpack("<i", f.read(4))[0]
        model_id = struct.unpack("<i", f.read(4))[0]

        width = struct.unpack("<Q", f.read(8))[0]
        height = struct.unpack("<Q", f.read(8))[0]

        params = struct.unpack(
            "<dddd",
            f.read(32)
        )

    # COLMAP camera model ID 1 = PINHOLE
    if model_id != 1:
        raise RuntimeError(
            f"Expected PINHOLE camera model (1), got {model_id}"
        )

    return width, height, params


# ============================================================
# CORRECT COLMAP QUATERNION -> ROTATION MATRIX
# ============================================================

def qvec_to_rotmat(q):

    qw, qx, qy, qz = q

    return np.array([
        [
            1 - 2 * (qy*qy + qz*qz),
            2 * (qx*qy - qz*qw),
            2 * (qx*qz + qy*qw)
        ],

        [
            2 * (qx*qy + qz*qw),
            1 - 2 * (qx*qx + qz*qz),
            2 * (qy*qz - qx*qw)
        ],

        [
            2 * (qx*qz - qy*qw),
            2 * (qy*qz + qx*qw),
            1 - 2 * (qx*qx + qy*qy)
        ]
    ], dtype=np.float64)


# ============================================================
# COLMAP IMAGE / CAMERA POSE READER
# ============================================================

def read_images(path):

    images = []

    with open(path, "rb") as f:

        count = struct.unpack("<Q", f.read(8))[0]

        for _ in range(count):

            image_id = struct.unpack(
                "<i",
                f.read(4)
            )[0]

            qvec = struct.unpack(
                "<dddd",
                f.read(32)
            )

            tvec = struct.unpack(
                "<ddd",
                f.read(24)
            )

            camera_id = struct.unpack(
                "<i",
                f.read(4)
            )[0]

            name_bytes = bytearray()

            while True:

                c = f.read(1)

                if c == b"\x00":
                    break

                name_bytes.extend(c)

            name = name_bytes.decode("utf-8")

            num_points = struct.unpack(
                "<Q",
                f.read(8)
            )[0]

            # x(double) + y(double) + point3D_id(int64)
            # = 24 bytes per observation.
            f.seek(
                num_points * 24,
                1
            )

            images.append({
                "id": image_id,
                "name": name,
                "camera_id": camera_id,
                "R": qvec_to_rotmat(qvec),
                "t": np.array(
                    tvec,
                    dtype=np.float64
                )
            })

    return images


# ============================================================
# LOAD TRUSTED SPARSE MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING TRUSTED SPARSE MODEL")
print("=" * 70)

sparse = o3d.io.read_point_cloud(
    SPARSE_PLY
)

if len(sparse.points) == 0:
    raise RuntimeError(
        "Sparse PLY contains no points."
    )

sparse_points = np.asarray(
    sparse.points
)

sparse_min = sparse_points.min(
    axis=0
)

sparse_max = sparse_points.max(
    axis=0
)

sparse_extent = (
    sparse_max -
    sparse_min
)

lower_bound = (
    sparse_min -
    BOUND_MARGIN * sparse_extent
)

upper_bound = (
    sparse_max +
    BOUND_MARGIN * sparse_extent
)

print(
    "Sparse points:",
    len(sparse_points)
)

print(
    "Sparse min:",
    sparse_min
)

print(
    "Sparse max:",
    sparse_max
)

print(
    "Allowed dense min:",
    lower_bound
)

print(
    "Allowed dense max:",
    upper_bound
)


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

print()
print("=" * 70)
print("LOADING CAMERA CALIBRATION")
print("=" * 70)

width, height, params = read_camera(
    os.path.join(
        MODEL_DIR,
        "cameras.bin"
    )
)

fx, fy, cx, cy = params

scale = TARGET_WIDTH / width

W = TARGET_WIDTH
H = int(
    round(height * scale)
)

K = np.array([
    [fx * scale, 0, cx * scale],
    [0, fy * scale, cy * scale],
    [0, 0, 1]
], dtype=np.float64)

print(
    "Original resolution:",
    width,
    "x",
    height
)

print(
    "Processing resolution:",
    W,
    "x",
    H
)

print(
    "fx:",
    K[0, 0]
)

print(
    "fy:",
    K[1, 1]
)

print(
    "cx:",
    K[0, 2]
)

print(
    "cy:",
    K[1, 2]
)


# ============================================================
# LOAD REGISTERED IMAGES
# ============================================================

images = read_images(
    os.path.join(
        MODEL_DIR,
        "images.bin"
    )
)

images.sort(
    key=lambda x: x["name"]
)

print()
print(
    "Registered images:",
    len(images)
)

if len(images) < 10:
    raise RuntimeError(
        "Too few registered images."
    )


# ============================================================
# SELECT 20 PAIRS ACROSS THE DATASET
# ============================================================

max_start = (
    len(images) -
    PAIR_STEP -
    1
)

pair_starts = np.linspace(
    0,
    max_start,
    MAX_PAIRS,
    dtype=int
)

pair_starts = np.unique(
    pair_starts
)

print(
    "Selected pairs:",
    len(pair_starts)
)


# ============================================================
# STEREO SGBM
#
# 512 disparities is intentionally chosen for RAM safety.
# ============================================================

stereo = cv2.StereoSGBM_create(

    minDisparity=0,

    numDisparities=512,

    blockSize=5,

    P1=8 * 3 * 25,

    P2=32 * 3 * 25,

    disp12MaxDiff=1,

    uniquenessRatio=8,

    speckleWindowSize=80,

    speckleRange=2,

    preFilterCap=31,

    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)


# ============================================================
# POINT STORAGE
# ============================================================

all_points = []
all_colors = []

successful_pairs = 0


# ============================================================
# PROCESS SELECTED PAIRS
# ============================================================

for pair_number, start in enumerate(
    pair_starts
):

    a = images[start]
    b = images[start + PAIR_STEP]

    print()
    print("=" * 70)
    print(
        f"PAIR {pair_number + 1}/{len(pair_starts)}"
    )

    print(
        "A:",
        a["name"]
    )

    print(
        "B:",
        b["name"]
    )


    # --------------------------------------------------------
    # READ IMAGES
    # --------------------------------------------------------

    path_a = os.path.join(
        IMAGE_DIR,
        a["name"]
    )

    path_b = os.path.join(
        IMAGE_DIR,
        b["name"]
    )

    img_a = cv2.imread(
        path_a,
        cv2.IMREAD_COLOR
    )

    img_b = cv2.imread(
        path_b,
        cv2.IMREAD_COLOR
    )

    if img_a is None or img_b is None:

        print(
            "Could not read images. Skipping."
        )

        continue


    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    img_a = cv2.resize(
        img_a,
        (W, H),
        interpolation=cv2.INTER_AREA
    )

    img_b = cv2.resize(
        img_b,
        (W, H),
        interpolation=cv2.INTER_AREA
    )

    gray_a = cv2.cvtColor(
        img_a,
        cv2.COLOR_BGR2GRAY
    )

    gray_b = cv2.cvtColor(
        img_b,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # CAMERA POSES
    #
    # COLMAP:
    # X_camera = R X_world + t
    # --------------------------------------------------------

    Ra = a["R"].copy()
    ta = a["t"].copy()

    Rb = b["R"].copy()
    tb = b["t"].copy()


    # --------------------------------------------------------
    # RELATIVE CAMERA TRANSFORM
    #
    # X_b = Rrel X_a + Trel
    # --------------------------------------------------------

    Rrel = Rb @ Ra.T

    Trel = (
        tb -
        Rrel @ ta
    ).reshape(3, 1)

    baseline = float(
        np.linalg.norm(Trel)
    )

    print(
        "Original baseline:",
        baseline
    )

    if baseline < 0.25:

        print(
            "Baseline too small. Skipping."
        )

        continue


    # --------------------------------------------------------
    # STEREO RECTIFICATION
    #
    # OpenCV's rectified camera geometry can handle the
    # relative pose directly. We don't force the physical
    # baseline sign by modifying the COLMAP poses.
    # --------------------------------------------------------

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(

        K,
        np.zeros(5),

        K,
        np.zeros(5),

        (W, H),

        Rrel,
        Trel,

        flags=cv2.CALIB_ZERO_DISPARITY,

        alpha=0
    )


    # --------------------------------------------------------
    # RECTIFICATION MAPS
    # --------------------------------------------------------

    map1x, map1y = cv2.initUndistortRectifyMap(

        K,
        np.zeros(5),

        R1,
        P1,

        (W, H),

        cv2.CV_32FC1
    )

    map2x, map2y = cv2.initUndistortRectifyMap(

        K,
        np.zeros(5),

        R2,
        P2,

        (W, H),

        cv2.CV_32FC1
    )


    rect_a = cv2.remap(
        gray_a,
        map1x,
        map1y,
        cv2.INTER_LINEAR
    )

    rect_b = cv2.remap(
        gray_b,
        map2x,
        map2y,
        cv2.INTER_LINEAR
    )

    color_a = cv2.remap(
        img_a,
        map1x,
        map1y,
        cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # DISPARITY
    # --------------------------------------------------------

    print(
        "Computing CPU disparity..."
    )

    disparity = stereo.compute(
        rect_a,
        rect_b
    ).astype(
        np.float32
    ) / 16.0


    # --------------------------------------------------------
    # BASIC DISPARITY FILTER
    # --------------------------------------------------------

    valid = (

        np.isfinite(disparity)

        &

        (disparity >= 2.0)

        &

        (disparity <= 512.0)
    )

    candidate_count = int(
        np.count_nonzero(valid)
    )

    print(
        "Candidate disparity pixels:",
        candidate_count
    )

    if candidate_count < 5000:

        print(
            "Too few valid disparity pixels."
        )

        continue


    # --------------------------------------------------------
    # REPROJECT DISPARITY TO 3D
    # --------------------------------------------------------

    points_rect_full = cv2.reprojectImageTo3D(
        disparity,
        Q
    )

    points_rect = (
        points_rect_full[valid]
    )

    colors = (
        color_a[valid][:, ::-1]
    )


    # --------------------------------------------------------
    # VALID 3D VALUES
    # --------------------------------------------------------

    good = (

        np.isfinite(points_rect).all(
            axis=1
        )

        &

        np.isfinite(
            points_rect[:, 2]
        )

        &

        (points_rect[:, 2] > 0)
    )

    points_rect = (
        points_rect[good]
    )

    colors = (
        colors[good]
    )


    print(
        "Positive-depth points:",
        len(points_rect)
    )

    if len(points_rect) < 1000:

        print(
            "Too few positive-depth points."
        )

        continue


    # --------------------------------------------------------
    # RECTIFIED CAMERA -> ORIGINAL CAMERA
    #
    # R1 maps original camera coordinates to rectified
    # coordinates.
    # --------------------------------------------------------

    points_camera = (
        R1.T @
        points_rect.T
    ).T


    # --------------------------------------------------------
    # ORIGINAL CAMERA -> WORLD
    #
    # X_world = R.T (X_camera - t)
    # --------------------------------------------------------

    points_world = (
        Ra.T @
        (
            points_camera -
            ta.reshape(1, 3)
        ).T
    ).T


    # --------------------------------------------------------
    # FINITE WORLD POINTS
    # --------------------------------------------------------

    good = np.isfinite(
        points_world
    ).all(
        axis=1
    )

    points_world = (
        points_world[good]
    )

    colors = (
        colors[good]
    )


    # --------------------------------------------------------
    # TRUSTED SPARSE GEOMETRY BOUND
    #
    # This eliminates catastrophic stereo depths.
    # --------------------------------------------------------

    good = np.all(

        (points_world >= lower_bound)

        &

        (points_world <= upper_bound),

        axis=1
    )

    points_world = (
        points_world[good]
    )

    colors = (
        colors[good]
    )


    print(
        "Accepted world points:",
        len(points_world)
    )

    if len(points_world) < 1000:

        print(
            "Too few geometrically valid points."
        )

        continue


    # --------------------------------------------------------
    # SAVE DEPTH MAP
    # --------------------------------------------------------

    depth_full = (
        points_rect_full[:, :, 2]
    )

    depth_valid = (

        np.isfinite(depth_full)

        &

        (depth_full > 0)
    )

    if np.any(depth_valid):

        values = depth_full[
            depth_valid
        ]

        lo = np.percentile(
            values,
            2
        )

        hi = np.percentile(
            values,
            98
        )

        normalized = np.clip(

            (
                depth_full - lo
            )
            /
            (
                hi - lo + 1e-8
            ),

            0,
            1
        )

        depth_png = (
            normalized * 255
        ).astype(
            np.uint8
        )

        base = os.path.splitext(
            a["name"]
        )[0]

        depth_path = os.path.join(
            DEPTH_DIR,
            base + "_depth.png"
        )

        cv2.imwrite(
            depth_path,
            depth_png
        )


    # --------------------------------------------------------
    # STORE POINTS
    # --------------------------------------------------------

    all_points.append(
        points_world
    )

    all_colors.append(
        colors
    )

    successful_pairs += 1


# ============================================================
# FINISH PAIRS
# ============================================================

print()
print("=" * 70)

print(
    "Successful pairs:",
    successful_pairs
)

if successful_pairs == 0:

    raise RuntimeError(
        "No stereo pair produced a valid reconstruction."
    )


# ============================================================
# COMBINE
# ============================================================

points = np.vstack(
    all_points
)

colors = np.vstack(
    all_colors
)

print(
    "Raw accepted points:",
    len(points)
)


# ============================================================
# OPEN3D CLOUD
# ============================================================

pcd = o3d.geometry.PointCloud()

pcd.points = (
    o3d.utility.Vector3dVector(
        points
    )
)

pcd.colors = (
    o3d.utility.Vector3dVector(
        colors.astype(
            np.float64
        ) / 255.0
    )
)


# ============================================================
# VOXEL DOWNSAMPLE
# ============================================================

print(
    "Voxel downsampling..."
)

pcd = pcd.voxel_down_sample(
    voxel_size=0.02
)

print(
    "After voxel downsampling:",
    len(pcd.points)
)


# ============================================================
# STATISTICAL OUTLIER REMOVAL
# ============================================================

if len(pcd.points) > 1000:

    print(
        "Removing statistical outliers..."
    )

    pcd, _ = (
        pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
    )


# ============================================================
# FINAL GEOMETRY
# ============================================================

final_points = np.asarray(
    pcd.points
)

if len(final_points) == 0:

    raise RuntimeError(
        "Final point cloud is empty."
    )

final_min = final_points.min(
    axis=0
)

final_max = final_points.max(
    axis=0
)

final_extent = (
    final_max -
    final_min
)


print()
print("=" * 70)
print("FINAL RECONSTRUCTION")
print("=" * 70)

print(
    "Final points:",
    len(final_points)
)

print(
    "Minimum:",
    final_min
)

print(
    "Maximum:",
    final_max
)

print(
    "Dimensions:",
    final_extent
)


# ============================================================
# SAVE FINAL PLY
# ============================================================

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
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Point cloud:",
    output
)

print(
    "Depth maps:",
    DEPTH_DIR
)
