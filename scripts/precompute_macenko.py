#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import pandas as pd

import staintools  # pip install staintools opencv-python

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


def find_img(root: Path, img_id: str) -> Path:
    for ext in IMG_EXTS:
        p = root / f"{img_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{img_id} image not found")


def choose_ref_id(img_dir: Path, train_df: pd.DataFrame, k: int = 32) -> str:
    """比較聰明地挑一張 reference patch

    步驟：
    1. 優先從 positive patches 抽樣 k 張，如果沒有 positive，就從全部裡面抽樣。
    2. 計算每張的亮度 mean、對比 std（用灰階）。
    3. 去掉幾乎全白 / 幾乎沒組織的 patch（std 太小）。
    4. 依照亮度排序，取中位數那張當 reference。
    """
    if "label" in train_df.columns and (train_df["label"] == 1).any():
        cand_df = train_df[train_df["label"] == 1].copy()
    else:
        cand_df = train_df.copy()

    cand_df = cand_df.sample(
        n=min(k, len(cand_df)), random_state=42, replace=False
    )

    stats = []
    for _, row in cand_df.iterrows():
        img_id = row["id"]
        p = find_img(img_dir, img_id)
        img = Image.open(p).convert("RGB")
        arr = np.array(img)  # HWC, uint8

        gray = arr.mean(axis=2)
        m = float(gray.mean())
        s = float(gray.std())
        stats.append((img_id, m, s))

    # 去掉對比太低的 (幾乎全白 / 幾乎沒 tissue)
    filtered = [s for s in stats if s[2] > 10.0]
    if not filtered:
        filtered = stats

    # 依亮度排序，取中位數那張
    filtered.sort(key=lambda x: x[1])  # by mean
    mid = filtered[len(filtered) // 2]
    ref_id = mid[0]
    print(f"[INFO] Chosen reference id: {ref_id} (mean={mid[1]:.1f}, std={mid[2]:.1f})")
    return ref_id


def build_macenko_normalizer(img_dir: Path, train_df: pd.DataFrame, ref_id: str = None):
    """
    建一個 Macenko stain normalizer：
    - 若 ref_id 為 None，就用 choose_ref_id() 自動在 positive patches 裡挑一張
    """
    if ref_id is None:
        ref_id = choose_ref_id(img_dir, train_df)
    else:
        print(f"[INFO] Using user-specified reference id: {ref_id}")

    ref_path = find_img(img_dir, ref_id)
    target = staintools.read_image(str(ref_path))  # uint8 RGB
    # 先做亮度標準化，比較穩定
    target = staintools.LuminosityStandardizer.standardize(target)

    normalizer = staintools.StainNormalizer(method="macenko")
    normalizer.fit(target)

    def _apply(img: np.ndarray) -> np.ndarray:
        """
        img: uint8 RGB (H, W, 3)
        回傳：Macenko normalized 的 uint8 RGB
        """
        # 幾乎全白 / 幾乎沒有組織：沒必要做 Macenko
        gray = img.mean(axis=2)
        if gray.mean() > 245 or gray.std() < 8:
            return img

        try:
            # 先做亮度標準化
            std_img = staintools.LuminosityStandardizer.standardize(img)
            out = normalizer.transform(std_img)

            if not np.isfinite(out).all():
                return img

            out = np.clip(out, 0, 255).astype(np.uint8)
            return out
        except Exception:
            # 任一奇怪錯誤 → 直接回傳原圖
            return img

    return _apply


def main(args):
    src_dir = Path(args.src_data_dir)
    dst_dir = Path(args.dst_data_dir)

    img_dir = src_dir / "train"
    dst_img_dir = dst_dir / "train"
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    # 讀 train.csv 來找 reference
    split_dir = Path(args.split_dir) if args.split_dir else src_dir / "splits/seed42_70_15_15"
    tr = pd.read_csv(split_dir / "train.csv")
    tr.columns = [c.strip().lower() for c in tr.columns]

    print("[INFO] Building Macenko normalizer (offline, HCD-tuned)...")
    stain_norm = build_macenko_normalizer(img_dir, tr, args.stain_ref_id)
    print("[INFO] Macenko normalizer ready.")

    # 把 train 資料夾底下的所有圖片都做一份 normalized copy
    img_paths = [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ]

    for p in tqdm(img_paths, desc="Offline Macenko (HCD-tuned)"):
        img = Image.open(p).convert("RGB")
        img_np = np.array(img)
        img_norm = stain_norm(img_np)
        img_out = Image.fromarray(img_norm)
        img_out.save(dst_img_dir / p.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_data_dir", required=True, help="原始 hcd 資料夾（裡面有 train/ 與 splits/）")
    ap.add_argument("--dst_data_dir", required=True, help="輸出 Macenko 後的資料夾，例如 hcd_macenko")
    ap.add_argument("--split_dir", default="", help="train/val/test 的 csv 位置（通常是 hcd/splits/...）")
    ap.add_argument("--stain_ref_id", type=str, default=None,
                    help="Macenko 的 reference patch id，可空白讓程式自選")

    args = ap.parse_args()
    main(args)
