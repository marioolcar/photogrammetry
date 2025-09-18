#!/usr/bin/env python3
"""
rail_3d_from_video.py

Pretvara .mp4 video vožnje vlaka u 3D prostor na dva načina:
  1) "colmap"  – Structure-from-Motion + Multi-View Stereo (PRECIZNIJE, zahtijeva COLMAP i ffmpeg)
  2) "midas"   – Monokularna dubina (BRŽE, bez vanjskih alata; točnost ovisi o sceni)

Ulaz: put do .mp4 videa
Izlaz: 3D oblak točaka i (opcionalno) mesh, spremljeni u output direktorij.

Primjeri pokretanja:
  python rail_3d_from_video.py input.mp4 --method colmap --fps 2 --out out_colmap
  python rail_3d_from_video.py input.mp4 --method midas  --fps 2 --out out_midas

Preduvjeti:
- Zajedničko: Python 3.9+, opencv-python, numpy, open3d
- Za "colmap": instalirani `ffmpeg` i `colmap` dostupni u PATH-u
- Za "midas": torch, torchvision, timm (model će se preuzeti automatski pri prvom pokretanju)

"""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import cv2

# Open3D je opcionalan za vizualizaciju i generiranje mesh-a
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
        # očisti postojeći sadržaj (ali pažljivo – briše cijeli dir)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


############################################################
# 1) EKSTRAKCIJA FRAMEOVA IZ VIDEA
############################################################

def extract_frames_ffmpeg(video_path: Path, frames_dir: Path, fps: float = 2.0, max_dim: int = 1280):
    """Ekstrahira frameove pomoću ffmpeg-a. Smanjuje rezoluciju na max_dim (širina ili visina)."""
    if not check_cmd("ffmpeg"):
        raise RuntimeError("ffmpeg nije pronađen u PATH-u. Instalirajte ffmpeg ili koristite --method midas bez ovog koraka.")

    # ensure_empty_dir(frames_dir)

    # scale filter: zadrži omjer, ograniči veću dimenziju na max_dim
    scale_filter = f"scale='if(gt(a,1),{max_dim},-2)':'if(gt(a,1),-2,{max_dim})'"

    # Izlazni pattern
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

    # ensure_empty_dir(work_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    # 2.1 Feature extractor (SIFT)
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", camera_model,
        # Pomoć kod video sekvenci – ovisno o snimanju može pomoći edge/affine…
        "--SiftExtraction.use_gpu", "1",
    ])

    # 2.2 Sequential matcher – za video sekvencu je bolje od exhaustive
    run([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
        "--SequentialMatching.overlap", "5",  # koliko susjednih frameova spajati
    ])

    # 2.3 Mapper (SfM) – rekonstrukcija kamere i sparse point cloud
    run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ])

    # Pronađi najveći model u sparse_dir (obično 0)
    models = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
    if not models:
        raise RuntimeError("COLMAP mapper nije proizveo modele.")
    model_dir = models[0]

    # 2.4 Undistort za MVS
    run([
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(model_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ])

    # 2.5 PatchMatch Stereo (dubine i normalni)
    run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0",
    ])

    # 2.6 Stereo fusion – gusti oblak točaka
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
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if "DPT" in model_type else transforms.small_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    return midas, transform, device


def depth_from_image(img_bgr, midas, transform, device):
    import torch
    from PIL import Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    depth = pred.cpu().numpy()
    # normalizacija u [0,1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def rgbd_to_point_cloud(rgb, depth_norm, fx=None, fy=None):
    """Pretvara RGB + normaliziranu dubinu u oblak točaka (Open3D)."""
    assert o3d is not None, "Open3D je potreban za point cloud."
    h, w = depth_norm.shape

    # Pretpostavi intrinzike kamere ako nisu dani
    if fx is None or fy is None:
        # empirijski: fokus ~ 1.2 * max(w, h)
        fx = fy = 1.2 * max(w, h)
    cx, cy = w / 2.0, h / 2.0

    depth_scaled = (depth_norm * 5.0).astype(np.float32)  # skala 0..~5m (podesivo)

    color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_scaled)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    # pretvori u metrički sustav (ovisno o skali iznad)
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])  # flip za Open3D koordinatni sustav
    return pcd


def midas_pipeline(video_path: Path, out_dir: Path, fps: float = 2.0, max_frames: int = 300) -> Path:
    """Ekstrahira frameove iz videa, računa dubinu MiDaS-om i spaja u jedan oblak točaka pomoću ICP-a.
       Vraća put do zajedničkog oblaka točaka (.ply).
    """
    if o3d is None:
        raise RuntimeError("Open3D je potreban za MiDaS pipeline.")

    # 1) Pročitaj frameove izravno iz videa (bez ffmpeg-a)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Ne mogu otvoriti video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(vid_fps / fps)))

    midas, transform, device = load_midas()

    accum_pcd = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if idx % step != 0:
            continue

        depth = depth_from_image(frame, midas, transform, device)
        pcd = rgbd_to_point_cloud(frame, depth)
        pcd = pcd.voxel_down_sample(voxel_size=0.05)

        if accum_pcd is None:
            accum_pcd = pcd
        else:
            # grubo poravnanje ICP-om
            reg = o3d.pipelines.registration.registration_icp(
                pcd, accum_pcd, 0.2,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            pcd.transform(reg.transformation)
            accum_pcd += pcd
            accum_pcd = accum_pcd.voxel_down_sample(voxel_size=0.05)
        frame_count += 1
        if frame_count >= max_frames:
            break

    cap.release()

    if accum_pcd is None:
        raise RuntimeError("Nije generiran nijedan oblak točaka.")

    pcd_path = out_dir / "midas_merged_point_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), accum_pcd)
    return pcd_path


############################################################
# 4) MESH IZ OBLAKA TOČAKA (OPEN3D)
############################################################

def mesh_from_point_cloud(ply_path: Path, out_mesh_path: Path, depth: int = 9, density: bool = True):
    assert o3d is not None, "Open3D je potreban za generiranje mesh-a."
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    print("[INFO] Poisson surface reconstruction…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    if density:
        densities = np.asarray(densities)
        # ukloni rijetke trokute
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_mesh_path), mesh)
    return out_mesh_path


############################################################
# MAIN
############################################################

def main():
    ap = argparse.ArgumentParser(description="Pretvori .mp4 u 3D prostor (COLMAP ili MiDaS)")
    ap.add_argument("video", type=Path, help="Put do .mp4 videa")
    ap.add_argument("--method", choices=["colmap", "midas"], default="colmap")
    ap.add_argument("--fps", type=float, default=2.0, help="Koliko frameova u sekundi uzeti iz videa")
    ap.add_argument("--out", type=Path, default=Path("output_3d"))
    ap.add_argument("--max_dim", type=int, default=1280, help="Maksimalna dimenzija framea kod ekstrakcije (colmap)")
    ap.add_argument("--make_mesh", action="store_true", help="Napravi mesh iz oblaka točaka")
    ap.add_argument("--mesh_depth", type=int, default=9, help="Poisson depth (mesh kvaliteta vs brzina)")
    ap.add_argument("--max_frames", type=int, default=300, help="Maks broj frameova (midas)")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "colmap":
        frames_dir = out_dir / "frames"
        work_dir = out_dir / "colmap"
        print("[STEP] Ekstrakcija frameova iz videa…")
        extract_frames_ffmpeg(args.video, frames_dir, fps=args.fps, max_dim=args.max_dim)
        print("[STEP] COLMAP rekonstrukcija…")
        fused_ply = colmap_pipeline(frames_dir, work_dir)
        print(f"[OK] Gusti oblak točaka: {fused_ply}")
        ply_path = fused_ply
    else:
        print("[STEP] MiDaS rekonstrukcija (monokularna dubina)…")
        ply_path = midas_pipeline(args.video, out_dir, fps=args.fps, max_frames=args.max_frames)
        print(f"[OK] Oblak točaka: {ply_path}")

    if args.make_mesh:
        if o3d is None:
            print("[WARN] Open3D nije dostupan – preskačem generiranje mesh-a")
        else:
            mesh_path = out_dir / ("mesh_colmap.ply" if args.method == "colmap" else "mesh_midas.ply")
            mesh_from_point_cloud(ply_path, mesh_path, depth=args.mesh_depth)
            print(f"[OK] Mesh spremljen: {mesh_path}")

    print("[DONE] Završeno.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
