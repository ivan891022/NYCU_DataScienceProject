#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HCD Macenko Stain Normalization (v2, EDA-guided)

- 使用 EDA 產生的 macenko_reference_id.txt 選出一張「典型」patch 當 reference
- 對 data_root/{train,test} 底下所有影像做離線 Macenko 染色標準化
- 存到 out_root/{train,test}，檔名不變（副檔名沿用原始）

設計重點：
- 忽略過於明亮（接近白色）的像素，只用有染色的組織做 SVD
- 將所有 patch 的 stain 濃度縮放到與 reference 相同的範圍
- 讓 brightness / saturation 的差異收斂，減少模型學顏色 shortcut
"""

import argparse
from pathlib import Path
import math
import multiprocessing as mp

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


# ----------------- Macenko utilities -----------------
def rgb2od(I: np.ndarray, Io: float = 255.0) -> np.ndarray:
    """RGB [0,255] -> optical density (OD)"""
    I = I.astype(np.float32)
    I[I <= 0] = 1.0  # avoid log(0)
    return -np.log(I / Io)


def od2rgb(OD: np.ndarray, Io: float = 255.0) -> np.ndarray:
    """
    OD -> RGB [0,255] uint8

    注意：Macenko 之後的線性重建有機會產生非常大的負值 / 正值，
    直接 exp 會 overflow，所以先把 OD clamp 到合理範圍再轉回 RGB。
    """
    # H&E 實際 OD 通常 <3，這裡保守一點設 0~5
    OD_clipped = np.clip(OD, 0.0, 5.0)

    I = Io * np.exp(-OD_clipped)
    I = np.clip(I, 0, Io)
    return I.astype(np.uint8)


def compute_stain_matrix_macenko(
    rgb: np.ndarray,
    alpha: float = 0.1,
    beta: float = 0.15,
    Io: float = 255.0,
) -> np.ndarray:
    """
    Macenko: estimate 3x2 stain matrix from一張 RGB 圖。

    alpha: percentile (0~1)，控制挑選兩端 angle 的極端點
    beta : OD threshold，用來排除太亮、沒染色的 pixel
    """
    h, w, _ = rgb.shape
    OD = rgb2od(rgb, Io=Io)
    OD = OD.reshape((-1, 3))

    # 只保留有染色的 pixel（任一 channel 超過 beta 即視為有染色）
    mask = np.any(OD > beta, axis=1)
    OD_tissue = OD[mask]

    # 如果整張圖幾乎都是白的，就回傳 None，讓外面 fallback 用 reference 的 W
    if OD_tissue.shape[0] < 0.01 * OD.shape[0]:
        return None

    # SVD on covariance of OD（等同於在 OD 空間做 PCA）
    cov = np.cov(OD_tissue.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh 回傳 eigenvalues 升冪，取最大兩個
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    V = eigvecs[:, :2]  # 3x2

    # 投影到 stain 子空間（2 維）
    projected = np.dot(OD_tissue, V)
    phi = np.arctan2(projected[:, 1], projected[:, 0])

    # 取兩端角度（alpha% 與 100-alpha%）
    min_phi = np.percentile(phi, alpha * 100)
    max_phi = np.percentile(phi, (1.0 - alpha) * 100)

    v_min = np.array([math.cos(min_phi), math.sin(min_phi)])
    v_max = np.array([math.cos(max_phi), math.sin(max_phi)])

    # 對應回 OD 空間 → 3x2 stain matrix
    HE = np.stack(
        [
            np.dot(V, v_min),  # 染色 1
            np.dot(V, v_max),  # 染色 2
        ],
        axis=1,
    )
    HE = HE / np.linalg.norm(HE, axis=0, keepdims=True)
    return HE.astype(np.float32)


def deconvolve(OD: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    給定 OD (3xN) 和 stain matrix W (3x2)，算出 stain 濃度 C (2xN)。
    """
    Wi = np.linalg.pinv(W)  # 2x3
    C = np.dot(Wi, OD)
    return C


def macenko_normalize(
    img_rgb: np.ndarray,
    W_ref: np.ndarray,
    C_ref_max: np.ndarray,
    alpha: float = 0.1,
    beta: float = 0.15,
    Io: float = 255.0,
) -> np.ndarray:
    """
    對一張圖做 Macenko normalize，mapping 到 reference 的 stain 濃度範圍。
    - W_ref: 參考圖的 3x2 stain matrix
    - C_ref_max: 參考圖兩個 stain 濃度的 99% 分位數（用來當目標 dynamic range）
    """
    h, w, _ = img_rgb.shape
    OD = rgb2od(img_rgb, Io=Io).reshape((-1, 3)).T  # 3xN

    # 估 target 自己的 stain matrix；如果失敗就直接用 reference 的
    W_tar = compute_stain_matrix_macenko(
        img_rgb, alpha=alpha, beta=beta, Io=Io
    )
    if W_tar is None:
        W_tar = W_ref

    # deconvolve
    C_tar = deconvolve(OD, W_tar)  # 2xN

    # 為了 robust，不用 max，而是 99% 分位數
    C_max_tar = np.percentile(C_tar, 99, axis=1)
    C_max_tar = np.maximum(C_max_tar, 1e-6)

    # scale 濃度到跟 reference 一樣的範圍
    scale = C_ref_max / C_max_tar
    C_scaled = C_tar * scale[:, None]

    # 用 reference 的 W 做 re-convolution
    OD_norm = np.dot(W_ref, C_scaled)  # 3xN
    OD_norm = OD_norm.T.reshape((h, w, 3))
    img_norm = od2rgb(OD_norm, Io=Io)
    return img_norm


# ----------------- IO helpers -----------------
def load_reference(
    data_root: Path,
    ref_id_file: Path,
    img_ext: str = ".tif",
    alpha: float = 0.1,
    beta: float = 0.15,
    Io: float = 255.0,
):
    """
    讀取 macenko_reference_id.txt，載入 train/<id>.ext，
    回傳 reference 的 W_ref(3x2) 和 C_ref_max(2,)。
    """
    with open(ref_id_file, "r") as f:
        ref_id = f.read().strip()

    ref_path = data_root / "train" / f"{ref_id}{img_ext}"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")

    rgb = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    W_ref = compute_stain_matrix_macenko(rgb, alpha=alpha, beta=beta, Io=Io)
    if W_ref is None:
        raise RuntimeError("Failed to compute stain matrix on reference image.")

    OD_ref = rgb2od(rgb, Io=Io).reshape((-1, 3)).T  # 3xN
    C_ref = deconvolve(OD_ref, W_ref)  # 2xN
    C_ref_max = np.percentile(C_ref, 99, axis=1)
    C_ref_max = np.maximum(C_ref_max, 1e-6)

    return ref_id, W_ref, C_ref_max


def list_images(folder: Path):
    paths = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def _worker_process_one(args):
    (
        src_path,
        dst_path,
        W_ref,
        C_ref_max,
        alpha,
        beta,
        Io,
    ) = args

    if dst_path.exists():
        return 0

    img_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return 0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_norm = macenko_normalize(
        img_rgb,
        W_ref=W_ref,
        C_ref_max=C_ref_max,
        alpha=alpha,
        beta=beta,
        Io=Io,
    )
    img_out = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dst_path), img_out)
    return 1


# ----------------- main -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        default="hcd",
        help="原始 HCD 資料根目錄（包含 train/, test/）",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="hcd_macenko_eda_v2",
        help="輸出根目錄（會在底下建立 train/, test/）",
    )
    ap.add_argument(
        "--ref_id_file",
        type=str,
        default="reports/hcd_color_eda/macenko_reference_id.txt",
        help="EDA 產生的 macenko_reference_id.txt 路徑（相對於 data_root）",
    )
    ap.add_argument(
        "--img_ext",
        type=str,
        default=".tif",
        help="影像副檔名（用在 reference, 其他格式會自動偵測）",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Macenko alpha (percentile, 0~1)",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=0.15,
        help="Macenko beta (OD threshold for tissue selection)",
    )
    ap.add_argument(
        "--Io",
        type=float,
        default=255.0,
        help="最大光強度 (通常 255)",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="CPU workers for multiprocessing",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="若已存在輸出檔，是否覆寫",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    ref_id_file = data_root / args.ref_id_file

    print(f"[INFO] data_root = {data_root.resolve()}")
    print(f"[INFO] out_root  = {out_root.resolve()}")
    print(f"[INFO] reference id file = {ref_id_file}")

    # ---- 1. 載入 reference & stain matrix ----
    ref_id, W_ref, C_ref_max = load_reference(
        data_root=data_root,
        ref_id_file=ref_id_file,
        img_ext=args.img_ext,
        alpha=args.alpha,
        beta=args.beta,
        Io=args.Io,
    )
    print(f"[INFO] Reference patch id = {ref_id}")
    print(f"[INFO] W_ref shape = {W_ref.shape}, C_ref_max = {C_ref_max}")

    # ---- 2. 對 train / test 跑離線 Macenko ----
    # 如果你現在完全不需要原始 Kaggle test，可以把這行改成 ["train"]
    subdirs = ["train", "test"]
    for sub in subdirs:
        src_dir = data_root / sub
        if not src_dir.exists():
            print(f"[WARN] {src_dir} not found, skip.")
            continue

        dst_dir = out_root / sub
        dst_dir.mkdir(parents=True, exist_ok=True)

        img_paths = list_images(src_dir)
        print(f"[INFO] [{sub}] found {len(img_paths)} images")

        jobs = []
        for p in img_paths:
            dst_p = dst_dir / p.name
            if dst_p.exists() and not args.overwrite:
                continue
            jobs.append(
                (
                    p,
                    dst_p,
                    W_ref,
                    C_ref_max,
                    args.alpha,
                    args.beta,
                    args.Io,
                )
            )

        print(f"[INFO] [{sub}] need to process {len(jobs)} images")

        if len(jobs) == 0:
            continue

        with mp.Pool(processes=args.num_workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(_worker_process_one, jobs),
                total=len(jobs),
                desc=f"Macenko {sub}",
            ):
                pass

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
