#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def split_stratified(df, train_ratio=0.7, val_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio - 0.85) < 1e-6
    test_ratio = 1 - (train_ratio + val_ratio)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1-train_ratio), random_state=seed)
    idx_tr, idx_tmp = next(sss1.split(df["id"], df["label"]))
    df_tr = df.iloc[idx_tr].copy(); df_tmp = df.iloc[idx_tmp].copy()

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio/(val_ratio+test_ratio), random_state=seed)
    idx_val, idx_te = next(sss2.split(df_tmp["id"], df_tmp["label"]))
    df_val = df_tmp.iloc[idx_val].copy(); df_te = df_tmp.iloc[idx_te].copy()
    return df_tr, df_val, df_te

def main(args):
    data_dir = Path(args.data_dir).expanduser().resolve()
    csv_path = data_dir/"train_labels.csv"
    out_dir = data_dir/f"splits/seed{args.seed}_70_15_15"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"id","label"}.issubset(df.columns)

    tr, val, te = split_stratified(df, 0.7, 0.15, seed=args.seed)
    tr.to_csv(out_dir/"train.csv", index=False)
    val.to_csv(out_dir/"val.csv", index=False)
    te.to_csv(out_dir/"test.csv", index=False)

    def stats(name, d): 
        n=len(d); p=int((d["label"]==1).sum()); neg=n-p
        print(f"{name:>5}: {n:,}  pos={p:,} ({p/n:.1%})  neg={neg:,} ({neg/n:.1%})")

    print(f"[OK] splits saved to: {out_dir}")
    stats("train", tr); stats(" val", val); stats("test", te)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="path to hcd/ (contains train/ & train_labels.csv)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); main(args)
