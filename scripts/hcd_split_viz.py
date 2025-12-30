#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize stratified splits for Kaggle HCD:
- Counts & pos ratios (train/val/test)
- Grouped bar chart (pos/neg by split)
- Donut charts (one per split)
- Sample grids of images per split & class

Outputs → {data_dir}/reports/data_preparation/splits_seed{seed}_70_15_15/
"""

import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

def find_img(img_dir: Path, img_id: str):
    for e in IMG_EXTS:
        p = img_dir / f"{img_id}{e}"
        if p.exists():
            return p
    return None

def fix_cols(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def ensure_out(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath

def make_donut(ax, counts, labels, title):
    # counts = [neg, pos] or [pos, neg] — we’ll label explicitly
    colors = None  # use matplotlib defaults
    wedges, _ = ax.pie(counts, startangle=90, wedgeprops=dict(width=0.45), colors=colors)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(wedges, labels, loc="center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)

def make_grid(img_paths, out_png, nrow=5, size=96):
    if len(img_paths) == 0:
        return
    imgs = []
    for p in img_paths[:nrow*nrow]:
        try:
            im = Image.open(p).convert("RGB").resize((size, size))
            imgs.append(np.asarray(im))
        except Exception:
            continue
    if not imgs:
        return
    imgs = np.stack(imgs, 0)
    H = W = nrow * size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    k = 0
    for r in range(nrow):
        for c in range(nrow):
            if k < len(imgs):
                canvas[r*size:(r+1)*size, c*size:(c+1)*size] = imgs[k]
            k += 1
    Image.fromarray(canvas).save(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="path to hcd/ (contains train/ & train_labels.csv)")
    ap.add_argument("--split_dir", default="", help="path to splits dir (default: {data_dir}/splits/seed{seed}_70_15_15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_per_class", type=int, default=25, help="images per class per split for grids")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    img_dir  = data_dir / "train"

    split_dir = Path(args.split_dir) if args.split_dir else (data_dir / f"splits/seed{args.seed}_70_15_15")
    assert split_dir.is_dir(), f"split_dir not found: {split_dir}"

    out_dir = ensure_out(data_dir / "reports" / "data_preparation" / f"splits_seed{args.seed}_70_15_15")

    # ---- read splits
    dfs = {}
    for s in ["train", "val", "test"]:
        df = fix_cols(pd.read_csv(split_dir / f"{s}.csv"))
        assert {"id", "label"} <= set(df.columns)
        dfs[s] = df

    # ---- stats table
    rows = []
    pos_ratio = {}
    for s, df in dfs.items():
        n = len(df)
        pos = int((df["label"] == 1).sum())
        neg = n - pos
        pos_ratio[s] = pos / n
        rows.append({"split": s, "n": n, "pos": pos, "neg": neg, "pos_ratio": round(pos_ratio[s], 4)})
    stats = pd.DataFrame(rows).sort_values("split")
    stats.to_csv(out_dir / "stats.csv", index=False)

    # ---- grouped bar (pos/neg per split)
    x = np.arange(len(dfs))
    width = 0.35
    pos_counts = [int((dfs[s]["label"] == 1).sum()) for s in ["train", "val", "test"]]
    neg_counts = [len(dfs[s]) - pos_counts[i] for i, s in enumerate(["train", "val", "test"])]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width/2, neg_counts, width, label="Negative")
    ax.bar(x + width/2, pos_counts, width, label="Positive")
    ax.set_xticks(x); ax.set_xticklabels(["train", "val", "test"])
    ax.set_ylabel("count"); ax.set_title("Class counts by split (stratified 70/15/15)")
    for i, (neg, pos) in enumerate(zip(neg_counts, pos_counts)):
        ax.text(i - width/2, neg + max(neg_counts)*0.01, f"{neg:,}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width/2, pos + max(pos_counts)*0.01, f"{pos:,}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bar_counts_by_split.png", dpi=160)
    plt.close(fig)

    # ---- donut (one per split)
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.4))
    for i, s in enumerate(["train", "val", "test"]):
        n = len(dfs[s]); pos = int((dfs[s]["label"] == 1).sum()); neg = n - pos
        make_donut(axs[i], [pos, neg], [f"pos {pos:,} ({pos/n:.1%})", f"neg {neg:,} ({neg/n:.1%})"], title=s)
    fig.suptitle("Pos/Neg ratio per split", y=1.05, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "donut_split_ratios.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- sample grids (per split per class)
    rng = random.Random(args.seed)
    for s, df in dfs.items():
        for klass, tag in [(1, "pos"), (0, "neg")]:
            ids = df.loc[df["label"] == klass, "id"].tolist()
            rng.shuffle(ids)
            picked = []
            for _id in ids[:args.sample_per_class*2]:  # oversample a bit to account for missing files
                p = find_img(img_dir, _id)
                if p is not None:
                    picked.append(p)
                if len(picked) >= args.sample_per_class:
                    break
            grid_path = out_dir / f"grid_{s}_{tag}.png"
            make_grid(picked, grid_path, nrow=int(np.ceil(np.sqrt(args.sample_per_class))), size=96)

    print(f"[OK] Saved figures & stats to: {out_dir}")
    print(stats.to_string(index=False))

if __name__ == "__main__":
    main()
