# Pixel-Ops Trial Round — Lighthouse 3D Reconstruction

## 1. Objective
Reconstruct the surveyed lighthouse structure from the provided multi-view UAV image dataset and provide the resulting 3D representation and an interactive visualization.

## 2. Dataset
The provided Lighthouse Survey dataset contains 200 JPG images captured from multiple viewpoints of the same real-world structure.

## 3. Reconstruction Pipeline
The implemented pipeline uses the following stages:

1. Feature extraction from the 200 input images using COLMAP with CPU-based SIFT.
2. Sequential feature matching with an image overlap of 10.
3. Structure-from-Motion (SfM) using COLMAP's mapper to estimate camera poses and triangulate 3D feature points.
4. Camera/image undistortion using the reconstructed camera model.
5. Export of the reconstructed sparse 3D point cloud to PLY format.
6. Depth-map experiments were implemented using OpenCV stereo processing on selected image pairs.
7. The final web visualization loads the verified COLMAP SfM point cloud using Three.js and PLYLoader.

## 4. Tools and Technologies
- Python
- COLMAP used as the reconstruction backend
- OpenCV
- Open3D
- Three.js
- HTML/CSS/JavaScript
- GitHub Pages

## 5. Final Output
The final verified reconstruction contains approximately 139,000 triangulated 3D points generated from the 200 survey images. The point cloud is provided in PLY format and is displayed through an interactive browser-based 3D viewer supporting rotation, panning and zooming.

## 6. Limitations
Dense multi-view stereo reconstruction was attempted using CPU-based methods because CUDA/GPU acceleration was unavailable on the development system. The experimental dense reconstructions produced significant geometric artifacts and were therefore not used as the final point cloud. The submitted final geometry is the verified COLMAP SfM sparse reconstruction. Consequently, surface completeness and density are lower than a successful dense MVS reconstruction.

## 7. Rule Adherence
COLMAP was used as a backend reconstruction tool rather than through its GUI. No ready-made point-cloud generation application was used as the final reconstruction. Open-source libraries were used for image processing, point-cloud handling and visualization. AI assistance was used for debugging and implementation support, while the pipeline and reconstruction stages are documented and explainable.
