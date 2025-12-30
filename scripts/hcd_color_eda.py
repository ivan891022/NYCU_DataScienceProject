#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def find_img(root: Path, img_id: str) -> Path:
    for ext in IMG_EXTS:
        p = root / f"{img_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{img_id} image not found under {root}")


def load_train_df(split_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(split_dir / "train.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"id", "label"}.issubset(df.columns):
        raise ValueError("train.csv must contain 'id' and 'label' columns")
    return df


def sample_stratified(df: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """在每個 label 內做均衡抽樣，總數約 n_samples"""
    labels = sorted(df["label"].unique())
    per_class = max(1, n_samples // len(labels))
    parts = []
    for y in labels:
        sub = df[df["label"] == y]
        take = min(len(sub), per_class)
        parts.append(sub.sample(n=take, random_state=seed))
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def compute_color_stats(img_root: Path, sample_df: pd.DataFrame, domain_name: str):
    """
    對一個資料來源（raw or macenko）計算：
      - 每張 patch 的 RGB / HSV 平均值
    回傳：DataFrame
    """
    rows = []
    train_dir = img_root / "train"
    print(f"[INFO] compute_color_stats for '{domain_name}' from {train_dir}")

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"{domain_name} patches"):
        img_id = row["id"]
        label = int(row["label"])
        p = find_img(train_dir, img_id)
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.float32)  # HWC, 0~255

        # RGB mean (0~1)
        rgb = arr / 255.0
        r_mean = float(rgb[..., 0].mean())
        g_mean = float(rgb[..., 1].mean())
        b_mean = float(rgb[..., 2].mean())

        # HSV mean (0~1)
        hsv = mcolors.rgb_to_hsv(rgb)
        h_mean = float(hsv[..., 0].mean())
        s_mean = float(hsv[..., 1].mean())
        v_mean = float(hsv[..., 2].mean())

        rows.append(
            {
                "id": img_id,
                "label": label,
                "domain": domain_name,
                "r": r_mean,
                "g": g_mean,
                "b": b_mean,
                "h": h_mean,
                "s": s_mean,
                "v": v_mean,
            }
        )

    return pd.DataFrame(rows)


def plot_hsv(stats: pd.DataFrame, out_dir: Path):
    domains = stats["domain"].unique()
    for dom in domains:
        sub = stats[stats["domain"] == dom]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, ch, title in zip(axes, ["h", "s", "v"], ["Hue mean", "Saturation mean", "Value mean"]):
            pos = sub[sub["label"] == 1][ch].values
            neg = sub[sub["label"] == 0][ch].values
            ax.hist(neg, bins=40, density=True, alpha=0.6, label="neg", edgecolor="none")
            ax.hist(pos, bins=40, density=True, alpha=0.6, label="pos", edgecolor="none")
            ax.set_title(f"{dom}: {title}")
            ax.set_xlabel("value")
            ax.set_ylabel("density")

        axes[0].legend()
        fig.tight_layout()
        out_path = out_dir / f"hsv_mean_{dom}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[INFO] saved {out_path}")


def plot_rgb(stats: pd.DataFrame, out_dir: Path):
    domains = stats["domain"].unique()
    for dom in domains:
        sub = stats[stats["domain"] == dom]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, ch, title in zip(axes, ["r", "g", "b"], ["R mean", "G mean", "B mean"]):
            pos = sub[sub["label"] == 1][ch].values
            neg = sub[sub["label"] == 0][ch].values
            ax.hist(neg, bins=40, density=True, alpha=0.6, label="neg", edgecolor="none")
            ax.hist(pos, bins=40, density=True, alpha=0.6, label="pos", edgecolor="none")
            ax.set_title(f"{dom}: {title}")
            ax.set_xlabel("value")
            ax.set_ylabel("density")

        axes[0].legend()
        fig.tight_layout()
        out_path = out_dir / f"rgb_mean_{dom}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[INFO] saved {out_path}")


def summarize_stats(stats: pd.DataFrame):
    print("\n========== Per-domain color summary ==========")
    for dom in sorted(stats["domain"].unique()):
        sub_dom = stats[stats["domain"] == dom]
        for lbl in [0, 1]:
            sub = sub_dom[sub_dom["label"] == lbl]
            if len(sub) == 0:
                continue
            m = sub[["h", "s", "v", "r", "g", "b"]].mean()
            print(f"[{dom}] label={lbl}  n={len(sub)}")
            print("  mean(H,S,V) = ", ", ".join(f"{x:.3f}" for x in m[["h", "s", "v"]].values))
            print("  mean(R,G,B) = ", ", ".join(f"{x:.3f}" for x in m[["r", "g", "b"]].values))
        print("----------------------------------------------")


def color_only_auc(stats: pd.DataFrame):
    print("\n========== Color-only 6D feature AUC (logistic, 5-fold) ==========")
    for dom in sorted(stats["domain"].unique()):
        sub = stats[stats["domain"] == dom]
        X = sub[["h", "s", "v", "r", "g", "b"]].values
        y = sub["label"].values.astype(int)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros_like(y, dtype=float)

        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr = y[tr_idx]

            clf = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
            )
            clf.fit(X_tr, y_tr)
            oof[val_idx] = clf.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y, oof)
        print(f"[{dom}] color-only AUC = {auc:.4f}")
    print("==============================================================\n")


def main(args):
    seed_all(args.seed)

    data_dir_raw = Path(args.data_dir_raw)
    split_dir = Path(args.split_dir) if args.split_dir else data_dir_raw / "splits/seed42_70_15_15"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 讀 train.csv & 抽樣
    train_df = load_train_df(split_dir)
    sample_df = sample_stratified(train_df, n_samples=args.n_samples, seed=args.seed)
    print(f"[INFO] Sampled {len(sample_df)} patches for EDA")

    # 2) raw 資料集的顏色統計
    all_stats = []
    all_stats.append(compute_color_stats(data_dir_raw, sample_df, domain_name="raw"))

    # 3) Macenko / 其他 normalized 資料集（可選）
    if args.data_dir_norm:
        norm_dir = Path(args.data_dir_norm)
        all_stats.append(compute_color_stats(norm_dir, sample_df, domain_name="normalized"))

    stats = pd.concat(all_stats, axis=0).reset_index(drop=True)

    # 4) summary + distribution plots + color-only AUC
    summarize_stats(stats)
    plot_hsv(stats, out_dir)
    plot_rgb(stats, out_dir)
    color_only_auc(stats)

    print(f"[INFO] EDA finished. Figures saved to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir_raw",
        required=True,
        help="原始 HCD 資料夾路徑（裡面要有 train/ 和 splits/）",
    )
    ap.add_argument(
        "--data_dir_norm",
        default="",
        help="離線 Macenko 或其他 normalized 資料夾（同樣有 train/），可留空只看 raw",
    )
    ap.add_argument(
        "--split_dir",
        default="",
        help="train/val/test 的 csv 資料夾（預設為 data_dir_raw/splits/seed42_70_15_15）",
    )
    ap.add_argument(
        "--n_samples",
        type=int,
        default=8000,
        help="用來做 EDA 的 patch 數量（總數，會在各 label 內做均衡抽樣）",
    )
    ap.add_argument("--out_dir", default="reports/color_eda", help="圖表輸出目錄")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)
