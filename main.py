#!/usr/bin/env python3
"""
rail_3d_from_video.py

Pretvara .mp4 video u 3D prostor na dva načina:
  1) "colmap"  – Structure-from-Motion + Multi-View Stereo (PRECIZNIJE, zahtijeva COLMAP i ffmpeg)
  2) "midas"   – Monokularna dubina (BRŽE, bez vanjskih alata; točnost ovisi o sceni)

Ulaz: put do .mp4 videa
Izlaz: 3D oblak točaka i (opcionalno) mesh, spremljeni u output direktorij.

Primjeri pokretanja:
    python main.py input.mp4 --method colmap --fps 2 --out out_colmap
    python main.py input.mp4 --method midas  --fps 2 --out out_midas

Preduvjeti:
    Nalaze se unutar requirements.txt
"""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm  # Dodano za progress bar

import numpy as np
import cv2

try:
    import open3d as o3d
except Exception:
    o3d = None

############################################################
# POMOĆNE FUNKCIJE
############################################################

def run(cmd, cwd=None):
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc.stdout


def check_cmd(name):
    return shutil.which(name) is not None


def ensure_empty_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


############################################################
# 1) EKSTRAKCIJA FRAMEOVA IZ VIDEA
############################################################

def extract_frames_ffmpeg(video_path: Path, frames_dir: Path, fps: float = 2.0, max_dim: int = 1280):
    """Ekstrahira frameove pomoću ffmpeg-a. Smanjuje rezoluciju na max_dim (širina ili visina)."""
    if not check_cmd("ffmpeg"):
        raise RuntimeError("ffmpeg nije pronađen u PATH-u. Instalirajte ffmpeg ili koristite --method midas bez ovog koraka.")

    scale_filter = f"scale='if(gt(a,1),{max_dim},-2)':'if(gt(a,1),-2,{max_dim})'"

    out_pattern = str(frames_dir / "%06d.jpg")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps},{scale_filter}",
        out_pattern,
    ]
    run(cmd)


############################################################
# 2) COLMAP PIPELINE (SfM + MVS)
############################################################

def colmap_pipeline(frames_dir: Path, work_dir: Path, camera_model: str = "SIMPLE_RADIAL") -> Path:
    """Pokreće minimalni COLMAP pipeline i vraća put do gustog oblaka točaka (fused.ply)."""
    if not check_cmd("colmap"):
        raise RuntimeError("COLMAP nije pronađen u PATH-u. Instalirajte COLMAP ili koristite --method midas.")

    db_path = work_dir / "database.db"
    images_dir = frames_dir
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP] COLMAP: Ekstrakcija značajki (SIFT)")
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", camera_model,
        "--SiftExtraction.use_gpu", "1",
    ])

    print("[STEP] COLMAP: Sekvencijalno uparivanje frameova")
    run([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
        "--SequentialMatching.overlap", "5",
    ])

    print("[STEP] COLMAP: Rekonstrukcija (mapper)")
    run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ])

    models = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
    if not models:
        raise RuntimeError("COLMAP mapper nije proizveo modele.")
    model_dir = models[0]

    print("[STEP] COLMAP: Undistort slika")
    run([
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(model_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ])

    print("[STEP] COLMAP: PatchMatch stereo (računanje dubina)")
    run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0",
    ])

    print("[STEP] COLMAP: Stereo fusion (generiranje oblaka točaka)")
    fused_ply = dense_dir / "fused.ply"
    run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_ply),
    ])

    if not fused_ply.exists():
        raise RuntimeError("Stereo fusion nije proizveo fused.ply")

    return fused_ply


############################################################
# 3) MIDAS PIPELINE (monokularna dubina)
############################################################

def load_midas(model_type: str = "DPT_Large"):
    import torch
    import torchvision.transforms as T

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    transform = T.Compose([
        T.Resize(384 if "small" in model_type.lower() else 512),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    return midas, transform, device


def depth_from_image(img_bgr, midas, transform, device):
    import torch
    from PIL import Image
    import numpy as np

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    inp = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = pred.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    return depth


def rgbd_to_point_cloud(rgb, depth_norm, fx=None, fy=None):
    assert o3d is not None, "Open3D je potreban za point cloud."
    h, w = depth_norm.shape

    if fx is None or fy is None:
        fx = fy = 1.2 * max(w, h)
    cx, cy = w / 2.0, h / 2.0

    depth_scaled = (depth_norm * 5.0).astype(np.float32)

    color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_scaled)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    return pcd


def midas_pipeline(video_path: Path, out_dir: Path, fps: float = 2.0, max_frames: int = 300) -> Path:
    if o3d is None:
        raise RuntimeError("Open3D je potreban za MiDaS pipeline.")

    cap = cv2.VideoCapture(str(video_path))
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_indices = list(range(0, frame_total, int(frame_total / max_frames)))
    frame_idx = 0
    accum_pcd = None
    midas, transform, device = load_midas()
    
    with tqdm(total=len(selected_indices), desc="[MiDaS] Obrada frameova") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(selected_indices):
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            target_frame = selected_indices[frame_idx]

            if current_frame < target_frame:
                continue
            elif current_frame > target_frame:
                frame_idx += 1
                continue

            depth = depth_from_image(frame, midas, transform, device)
            pcd = rgbd_to_point_cloud(frame, depth)
            pcd = pcd.voxel_down_sample(voxel_size=0.05)

            if accum_pcd is None:
                accum_pcd = pcd
            else:
                accum_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                reg = o3d.pipelines.registration.registration_icp(
                    pcd, accum_pcd, 0.2,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                pcd.transform(reg.transformation)
                accum_pcd += pcd
                accum_pcd = accum_pcd.voxel_down_sample(voxel_size=0.05)

            frame_idx += 1
            pbar.update(1)

    return accum_pcd


############################################################
# MAIN FUNC
############################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Put do video datoteke.")
    parser.add_argument("--out", type=Path, default="output", help="Izlazna mapa")
    parser.add_argument("--method", choices=["colmap", "midas"], default="colmap", help="Metoda rekonstrukcije")
    parser.add_argument("--fps", type=float, default=2.0, help="FPS za ekstrakciju frameova")
    parser.add_argument("--max_frames", type=int, default=300, help="Maksimalni broj frameova (samo za midas)")

    args = parser.parse_args()

    video_path = args.video
    out_dir = args.out

    ensure_empty_dir(out_dir)

    if args.method == "colmap":
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        extract_frames_ffmpeg(video_path, frames_dir, fps=args.fps)
        fused_ply = colmap_pipeline(frames_dir, out_dir)
        print(f"[INFO] Kolmap 3D model spremljen u: {fused_ply}")

    elif args.method == "midas":
        pcd = midas_pipeline(video_path, out_dir, fps=args.fps)
        output_pcd = out_dir / "midas_output.ply"
        o3d.io.write_point_cloud(str(output_pcd), pcd)
        print(f"[INFO] MiDaS 3D model spremljen u: {output_pcd}")


if __name__ == "__main__":
    main()
