#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class Balance Plot for Kaggle Histopathologic Cancer Detection (PCam/HCD)
- Input : <data_dir>/train_labels.csv  (columns: id,label)
- Output: <out_dir>/class_balance_donut.png, class_balance_bar.png, class_balance_summary.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # 避免伺服器無顯示環境出錯

def fix_cols(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def plot_donut(pos, neg, out_png, title="Train Class Balance"):
    total = pos + neg
    sizes = [pos, neg]
    labels = [f"Positive\n{pos:,} ({pos/total:.1%})",
              f"Negative\n{neg:,} ({neg/total:.1%})"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _ = ax.pie(
        sizes,
        startangle=90, counterclock=False,
        labels=None,            # 我們用 legend 顯示文字
        wedgeprops=dict(width=0.38, edgecolor="white")
    )
    # 置中顯示總數
    ax.text(0, 0, f"N = {total:,}", ha="center", va="center",
            fontsize=16, fontweight="bold")

    ax.set(aspect="equal", title=title)
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.03, 0.5),
              frameon=False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_bar(pos, neg, out_png, title="Train Class Balance (Counts)"):
    total = pos + neg
    cats = ["Negative", "Positive"]
    vals = [neg, pos]
    perc = [neg/total, pos/total]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars = ax.bar(cats, vals)
    ax.set_ylabel("Count")
    ax.set_title(title)

    # 在柱上方加百分比
    for b, p in zip(bars, perc):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{p:.1%}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    # 顯示網格更易讀
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="資料夾，裡面要有 train_labels.csv")
    ap.add_argument("--out_dir", default="", help="輸出資料夾，預設為 <data_dir>/reports/data_understanding/")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    labels_csv = data_dir / "train_labels.csv"
    assert labels_csv.exists(), f"{labels_csv} 不存在"

    out_dir = Path(args.out_dir) if args.out_dir else (data_dir / "reports" / "data_understanding")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fix_cols(pd.read_csv(labels_csv))
    assert {"id", "label"} <= set(df.columns), "train_labels.csv 必須包含欄位 id,label"

    n_total = len(df)
    n_pos = int(df["label"].sum())
    n_neg = n_total - n_pos
    pos_ratio = n_pos / n_total

    # 存 summary
    pd.DataFrame([{
        "n_total": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "pos_ratio": pos_ratio
    }]).to_csv(out_dir / "class_balance_summary.csv", index=False)

    # 兩張圖
    plot_donut(n_pos, n_neg, out_dir / "class_balance_donut.png")
    plot_bar(n_pos, n_neg, out_dir / "class_balance_bar.png")

    print(f"[OK] Total={n_total:,}  Pos={n_pos:,} ({pos_ratio:.2%})  Neg={n_neg:,}")
    print(f"[Saved] {out_dir/'class_balance_donut.png'}")
    print(f"[Saved] {out_dir/'class_balance_bar.png'}")
    print(f"[Saved] {out_dir/'class_balance_summary.csv'}")

if __name__ == "__main__":
    main()
