#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HCD 色彩增強 EDA

- 用法示例：

  原始 HCD：
    python scripts/hcd_color_aug_eda.py \
      --data_dir hcd \
      --split_csv hcd/splits/seed42_70_15_15/train.csv \
      --out_prefix reports/hcd_raw

  Macenko v2：
    python scripts/hcd_color_aug_eda.py \
      --data_dir hcd_macenko_eda_v2 \
      --split_csv hcd/splits/seed42_70_15_15/train.csv \
      --out_prefix reports/hcd_macenko_v2

- 會輸出：
    <out_prefix>_color_stats.csv   : 每張 sample 的顏色統計
    <out_prefix>_brightness.png    : brightness 分佈 (不同 aug)
    <out_prefix>_contrast.png      : contrast 分佈
    <out_prefix>_rb_ratio.png      : R/B ratio 分佈
"""

import argparse
import os
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def find_img(root: Path, img_id: str) -> Path:
    for ext in IMG_EXTS:
        p = root / f"{img_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{img_id} image not found under {root}")


# ---------- 定義幾組「純顏色增強」pipeline（不含 Normalize/ToTensor） ----------
def build_color_pipelines():
    """
    回傳一個 dict：name -> albumentations.Compose
    這裡只做「顏色」相關的操作，不做幾何，方便單獨觀察色彩變化
    """
    pipelines = {}

    # 0) 完全不動顏色
    pipelines["none"] = A.Compose([])

    # 1) mild_bc：亮度 / 對比 +-10%
    pipelines["mild_bc"] = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=1.0,
            )
        ]
    )

    # 2) mild_bc_gamma：亮度 / 對比 + 輕微 gamma
    pipelines["mild_bc_gamma"] = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=1.0,
            ),
            A.RandomGamma(
                gamma_limit=(90, 110),
                p=1.0,
            ),
        ]
    )

    # 3) bc_rgbshift：亮度 / 對比 + 很小的 RGB shift
    pipelines["bc_rgbshift"] = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=1.0,
            ),
            A.RGBShift(
                r_shift_limit=3,
                g_shift_limit=3,
                b_shift_limit=3,
                p=1.0,
            ),
        ]
    )

    # 4) strong_hsv（故意比較兇，當反例看用）
    pipelines["strong_hsv"] = A.Compose(
        [
            A.HueSaturationValue(
                hue_shift_limit=0.05,   # 大約 18 度
                sat_shift_limit=0.30,
                val_shift_limit=0.30,
                p=1.0,
            )
        ]
    )

    return pipelines


def compute_color_stats(img_rgb: np.ndarray) -> dict:
    """
    img_rgb: uint8, HWC, RGB
    回傳 brightness / contrast / R,G,B mean,std / R/B ratio
    """
    img = img_rgb.astype(np.float32) / 255.0

    # 灰階
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0

    brightness = float(gray.mean())
    contrast = float(gray.std() + 1e-8)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r_mean, g_mean, b_mean = float(r.mean()), float(g.mean()), float(b.mean())
    r_std, g_std, b_std = float(r.std()), float(g.std()), float(b.std())

    rb_ratio = float(r_mean / (b_mean + 1e-6))  # 粗略 H/E 比例感覺一下

    return {
        "brightness": brightness,
        "contrast": contrast,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "r_std": r_std,
        "g_std": g_std,
        "b_std": b_std,
        "rb_ratio": rb_ratio,
    }


def main(args):
    seed_all(args.seed)

    data_dir = Path(args.data_dir)
    img_root = data_dir / "train"

    df = pd.read_csv(args.split_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    # 只抽一部分 sample 來做 EDA（避免太慢）
    if args.n_samples > 0 and args.n_samples < len(df):
        df = df.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)

    color_pipelines = build_color_pipelines()

    rows = []

    print(f"[INFO] EDA samples: {len(df)} patches")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row["id"]
        label = int(row["label"])

        img_path = find_img(img_root, img_id)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)  # uint8 HWC

        # 對每個顏色 pipeline 做多次 sample（增加穩定度）
        for aug_name, pipeline in color_pipelines.items():
            for k in range(args.n_repeats):
                if len(pipeline.transforms) > 0:
                    aug_img = pipeline(image=img)["image"]
                else:
                    aug_img = img

                stats = compute_color_stats(aug_img)
                stats.update(
                    {
                        "id": img_id,
                        "label": label,
                        "aug": aug_name,
                        "repeat": k,
                    }
                )
                rows.append(stats)

    stats_df = pd.DataFrame(rows)
    out_csv = f"{args.out_prefix}_color_stats.csv"
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved stats to: {out_csv}")

    # ---------- 畫幾張關鍵圖 ----------
    sns.set(style="whitegrid")

    def save_boxplot(metric: str, fname: str):
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=stats_df,
            x="aug",
            y=metric,
            # 你也可以用 hue="label" 分陽性/陰性
        )
        plt.title(f"{metric} by color augmentation")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

    save_boxplot("brightness", f"{args.out_prefix}_brightness.png")
    save_boxplot("contrast", f"{args.out_prefix}_contrast.png")
    save_boxplot("rb_ratio", f"{args.out_prefix}_rb_ratio.png")

    print(f"[INFO] Saved plots with prefix: {args.out_prefix}_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="hcd 或 hcd_macenko_eda_v2")
    ap.add_argument(
        "--split_csv",
        required=True,
        help="例如 hcd/splits/seed42_70_15_15/train.csv",
    )
    ap.add_argument(
        "--out_prefix",
        required=True,
        help="輸出檔名前綴，例如 reports/hcd_raw",
    )
    ap.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="從 train.csv 抽多少 patch 來做 EDA（0 或負數代表用全部）",
    )
    ap.add_argument(
        "--n_repeats",
        type=int,
        default=3,
        help="同一張圖對每個 aug 重複幾次（不同隨機 seed）",
    )
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)
