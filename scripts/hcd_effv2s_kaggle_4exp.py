#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kaggle Histopathologic Cancer Detection
EfficientNetV2-S (ImageNet-1k) + 幾何增強 (+/- 顏色增強) + DDP

資料結構：
    hcd/
        train/                  (Kaggle original train images)
        test/                   (Kaggle original test images)
        train_labels.csv
        sample_submission.csv

    hcd_macenko_eda_v2/
        train/                  (Macenko 離線染色標準化 train)
        test/                   (Macenko 離線染色標準化 test)

這支 script 支援 4 種實驗：
    1) raw：              data_dir=hcd                    , 不加 --use_color_aug
    2) raw + color aug：  data_dir=hcd                    , 加   --use_color_aug
    3) macenko：          data_dir=hcd_macenko_eda_v2     , 不加 --use_color_aug
    4) macenko + color：  data_dir=hcd_macenko_eda_v2     , 加   --use_color_aug

流程：
    1) 從 --labels_csv 讀 train_labels.csv（預設 hcd/train_labels.csv）
    2) 做 stratified train/val split（val_frac，預設 0.15）
    3) 在 train 訓練，在 val 上選 AUROC 最好的 epoch 存 ckpt
    4) 讀回 best ckpt：
        - 在 val 上報 AUROC / F1 / thr
        - 用 data_dir/test 做推論，改寫 --sample_csv，輸出 submission

注意：
    * 顏色增強對 raw / macenko 設計不同：raw 比較強、macenko 比較溫和
    * AMP 使用 torch.cuda.amp (GradScaler + autocast)
"""

import os
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

import albumentations as A
import albumentations.pytorch as AP
import timm

from torch.cuda.amp import GradScaler, autocast

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


# ----------------- utils -----------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.6, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        loss = (
            (self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets))
            * (1.0 - pt) ** self.gamma
            * bce
        )
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def init_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def gather_lists_py(obj_list):
    """在 DDP 下把各 GPU 的 list 合併成一個大 list"""
    if not dist.is_initialized():
        return obj_list
    world = dist.get_world_size()
    bufs = [None for _ in range(world)]
    dist.all_gather_object(bufs, obj_list)
    out = []
    for b in bufs:
        out.extend(b)
    return out


# ----------------- data / aug -----------------
def find_img(root: Path, img_id: str) -> Path:
    for ext in IMG_EXTS:
        p = root / f"{img_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{img_id} image not found under {root}")


def build_train_transform(img_size: int, use_color_aug: bool, is_macenko: bool):
    """
    幾何增強（所有實驗都開）+ 可選的顏色增強。

    幾何：
        - H/V flip
        - 小角度 rotate
        - Resize 到 img_size
    顏色：
        - raw: 比較強的 BC + gamma + 小 RGBShift
        - macenko: 已標準化，顏色增強更溫和（小 BC + 小 gamma）
    """
    geo_transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ),
        A.Resize(img_size, img_size),
    ]

    # 顏色增強 for 原始 HCD
    color_tf_raw = A.OneOf(
        [
            A.Compose(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.12,
                        contrast_limit=0.12,
                        p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                ]
            ),
            A.Compose(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.08,
                        contrast_limit=0.08,
                        p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                    A.RGBShift(
                        r_shift_limit=4,
                        g_shift_limit=4,
                        b_shift_limit=4,
                        p=0.5,
                    ),
                ]
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.04,
                contrast_limit=0.04,
                p=1.0,
            ),
        ],
        p=0.9,
    )

    # 顏色增強 for Macenko（更保守）
    color_tf_macenko = A.OneOf(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.06,
                contrast_limit=0.06,
                p=1.0,
            ),
            A.Compose(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.04,
                        contrast_limit=0.04,
                        p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=(96, 104), p=1.0),
                ]
            ),
            A.RandomGamma(gamma_limit=(95, 105), p=1.0),
        ],
        p=0.8,
    )

    tf_list = list(geo_transforms)
    if use_color_aug:
        tf_list.append(color_tf_macenko if is_macenko else color_tf_raw)

    tf_list.extend(
        [
            A.Normalize(
                mean=(0.67, 0.45, 0.69),
                std=(0.23, 0.21, 0.22),
            ),
            AP.ToTensorV2(),
        ]
    )
    return A.Compose(tf_list)


class HCDDataset(Dataset):
    """有 label 的 train / val"""

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        train: bool,
        img_size: int,
        use_color_aug: bool,
        is_macenko: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train
        self.img_size = img_size

        if train:
            self.tf = build_train_transform(
                img_size=self.img_size,
                use_color_aug=use_color_aug,
                is_macenko=is_macenko,
            )
        else:
            self.tf = A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.Normalize(
                        mean=(0.67, 0.45, 0.69),
                        std=(0.23, 0.21, 0.22),
                    ),
                    AP.ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img_path = find_img(self.img_dir, r["id"])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        label = int(r["label"])
        x = self.tf(image=img)["image"]
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


class HCDKaggleTestDataset(Dataset):
    """Kaggle 官方 test 集（沒有 label，只回傳 id 和影像）"""

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        img_size: int,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.tf = A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=(0.67, 0.45, 0.69),
                    std=(0.23, 0.21, 0.22),
                ),
                AP.ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img_id = self.df.loc[i, "id"]
        img_path = find_img(self.img_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        x = self.tf(image=img)["image"]
        return x, img_id


# ----------------- model / metric -----------------
def build_model(
    name: str = "tf_efficientnetv2_s",  # ImageNet-1k 版本
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.2,
):
    model = timm.create_model(
        name,
        pretrained=True,
        in_chans=3,
        num_classes=1,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    return model


def class_pos_weight(labels: np.ndarray) -> float:
    n = float(len(labels))
    pos = float(labels.sum())
    neg = n - pos
    return neg / max(pos, 1.0)


def best_thr_metrics(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p * r / (p + r + 1e-9)

    if len(t) == 0:
        thr = 0.5
        y_pred = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1_best = 2 * prec * rec / (prec + rec + 1e-9)
        return thr, f1_best, prec, rec

    f1_for_t = f1[1:]
    best_rel = int(np.nanargmax(f1_for_t))
    idx = best_rel + 1
    thr = float(t[best_rel])
    prec = float(p[idx])
    rec = float(r[idx])
    f1_best = float(f1[idx])
    return thr, f1_best, prec, rec


@torch.no_grad()
def predict_tta(model, x, use_tta: bool):
    # 一般建議不要開 TTA；這裡只是保留選項
    if not use_tta:
        return torch.sigmoid(model(x))

    views = []
    views.append(x)
    views.append(torch.flip(x, dims=[3]))
    views.append(torch.flip(x, dims=[2]))

    x90 = torch.rot90(x, k=1, dims=[2, 3])
    x180 = torch.rot90(x, k=2, dims=[2, 3])
    x270 = torch.rot90(x, k=3, dims=[2, 3])
    views.extend([x90, x180, x270])

    views.append(torch.flip(x90, dims=[3]))
    views.append(torch.flip(x90, dims=[2]))

    logits_list = [model(v) for v in views]
    logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return torch.sigmoid(logits)


@torch.no_grad()
def evaluate(model, dl, device, use_tta: bool = False):
    model.eval()
    probs_all, targs_all = [], []

    for x, y in dl:
        x = x.to(device, non_blocking=True)

        pr_tensor = predict_tta(model, x, use_tta=use_tta)
        pr = pr_tensor.squeeze(1).cpu().numpy().tolist()
        tg = y.squeeze(1).cpu().numpy().tolist()

        probs_all.extend(pr)
        targs_all.extend(tg)

    probs_all = gather_lists_py(probs_all)
    targs_all = gather_lists_py(targs_all)

    y_prob = np.array(probs_all)
    y_true = np.array(targs_all, dtype=int)
    auroc = roc_auc_score(y_true, y_prob)
    thr, f1, prec, rec = best_thr_metrics(y_true, y_prob)
    return auroc, f1, thr, prec, rec


@torch.no_grad()
def predict_for_submission(model, dl, device, use_tta: bool = False):
    """對 Kaggle test 做推論，回傳 id list 與 prob list"""
    model.eval()
    ids_all, probs_all = [], []

    for x, ids in dl:
        x = x.to(device, non_blocking=True)

        pr_tensor = predict_tta(model, x, use_tta=use_tta)
        pr = pr_tensor.squeeze(1).cpu().numpy().tolist()

        probs_all.extend(pr)
        ids_all.extend(list(ids))

    probs_all = gather_lists_py(probs_all)
    ids_all = gather_lists_py(ids_all)

    return ids_all, probs_all


# ----------------- main train loop -----------------
def main(args):
    seed_all(args.seed)
    local_rank, rank, world = init_ddp()
    device = torch.device(f"cuda:{local_rank}")

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "train"
    is_macenko = "macenko" in str(data_dir).lower()

    # ---- 讀 train_labels.csv，做 stratified train/val split ----
    labels_path = Path(args.labels_csv)
    df_all = pd.read_csv(labels_path)
    df_all.columns = [c.strip().lower() for c in df_all.columns]
    assert "id" in df_all.columns and "label" in df_all.columns

    train_df, val_df = train_test_split(
        df_all,
        test_size=args.val_frac,
        stratify=df_all["label"],
        random_state=args.seed,
    )

    if is_main():
        pos_all = df_all["label"].mean()
        pos_tr = train_df["label"].mean()
        pos_val = val_df["label"].mean()
        print(
            f"[DATA] total={len(df_all)} (pos={pos_all:.4f})  "
            f"train={len(train_df)} (pos={pos_tr:.4f})  "
            f"val={len(val_df)} (pos={pos_val:.4f})"
        )
        print(f"[DATA] data_dir={data_dir}, is_macenko={is_macenko}")

    # Dataset / Dataloader
    train_set = HCDDataset(
        train_df,
        img_dir,
        train=True,
        img_size=args.img_size,
        use_color_aug=args.use_color_aug,
        is_macenko=is_macenko,
    )
    val_set = HCDDataset(
        val_df,
        img_dir,
        train=False,
        img_size=args.img_size,
        use_color_aug=False,
        is_macenko=is_macenko,
    )

    train_samp = DistributedSampler(train_set, shuffle=True, drop_last=False)
    val_samp = DistributedSampler(val_set, shuffle=False, drop_last=False)

    train_dl = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_samp,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
    val_dl = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_samp,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Model
    model = build_model(
        name=args.model,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    model = model.to(memory_format=torch.channels_last)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if args.eval_only:
        raise NotImplementedError(
            "--eval_only 模式暫時沒做，如需我可以再幫你加"
        )

    # ----- loss -----
    if args.use_focal:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        if is_main():
            print("[INFO] Using FocalLoss")
    else:
        base_pw = class_pos_weight(train_df["label"].values)
        pw = base_pw * args.pos_weight_scale
        pos_w = torch.tensor(pw, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        if is_main():
            print(
                f"[INFO] Using BCEWithLogitsLoss, base_pos_weight={base_pw:.3f}, "
                f"scale={args.pos_weight_scale:.3f}, final_pos_weight={pw:.3f}"
            )

    # LR 線性縮放
    effective_accum = max(args.grad_accum_steps, 1)
    global_bsz = args.batch_size * world * effective_accum

    if args.lr_scale_ref:
        base_lr = args.base_lr * (global_bsz / args.lr_scale_ref)
    else:
        base_lr = args.base_lr

    optim = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=base_lr * 0.1
    )
    scaler = GradScaler(enabled=args.amp)

    if is_main():
        print(
            f"[INFO] world={world}  perGPU-bsz={args.batch_size}  "
            f"grad_accum={effective_accum}  "
            f"effective-global-bsz={global_bsz}  lr={base_lr:.2e}"
        )
        print(f"[INFO] loss={'Focal' if args.use_focal else 'BCEw'}")
        print(f"[INFO] TTA for eval/submission: {args.tta}")
        print(
            f"[INFO] color_aug={args.use_color_aug}  "
            f"(raw vs macenko handled automatically)"
        )
        print("[INFO] norm_mean=(0.67, 0.45, 0.69)  norm_std=(0.23, 0.21, 0.22)")

    best = {"auroc": -1.0, "f1": -1.0, "thr": 0.5, "epoch": -1}
    ckpt_path = Path(args.out_dir) / f"{args.model}_best.pth"

    # ---------- training ----------
    for ep in range(1, args.epochs + 1):
        model.train()
        train_samp.set_epoch(ep)
        pbar = tqdm(
            train_dl,
            disable=not is_main(),
            desc=f"Epoch {ep}/{args.epochs}",
        )

        optim.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            y = y.to(device)

            with autocast(enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)
                loss = loss / effective_accum

            scaler.scale(loss).backward()

            do_step = (
                (step + 1) % effective_accum == 0
                or (step + 1) == len(train_dl)
            )
            if do_step:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            if is_main():
                pbar.set_postfix(
                    loss=float(loss.detach().mean()) * effective_accum
                )

        sched.step()

        # val
        au, f1, thr, prec, rec = evaluate(
            model, val_dl, device, use_tta=args.tta
        )
        if is_main():
            print(
                f"  -> val AUROC={au:.4f}  F1*={f1:.4f}  "
                f"thr={thr:.3f}  P={prec:.4f}  R={rec:.4f}"
            )

        if au > best["auroc"]:
            best.update({"auroc": au, "f1": f1, "thr": thr, "epoch": ep})
            if is_main():
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.module.state_dict(), **best}, ckpt_path)

    # ---------- reload best ckpt ----------
    if dist.is_initialized():
        dist.barrier()

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(state["model"])
    else:
        if is_main():
            print("[WARN] best checkpoint not found, using last-epoch weights.")

    au, f1, thr, prec, rec = evaluate(
        model, val_dl, device, use_tta=args.tta
    )
    if is_main():
        print(
            f"[VAL(best)] AUROC={au:.4f}  F1*={f1:.4f}  thr={thr:.3f}  "
            f"P={prec:.4f}  R={rec:.4f}"
        )
        if best["epoch"] > 0:
            print(
                f"[BEST(val)] epoch={best['epoch']}  "
                f"AUROC={best['auroc']:.4f}  F1*={best['f1']:.4f}  thr={best['thr']:.3f}"
            )

    # ---------- 產生 Kaggle submission ----------
    if dist.is_initialized():
        dist.barrier()

    sample_sub_path = Path(args.sample_csv)
    sample_df = pd.read_csv(sample_sub_path)
    test_ids_df = sample_df[["id"]].copy()
    kaggle_img_dir = data_dir / "test"

    kaggle_set = HCDKaggleTestDataset(
        test_ids_df,
        kaggle_img_dir,
        img_size=args.img_size,
    )
    kaggle_samp = DistributedSampler(
        kaggle_set, shuffle=False, drop_last=False
    )
    kaggle_dl = DataLoader(
        kaggle_set,
        batch_size=args.batch_size,
        sampler=kaggle_samp,
        num_workers=args.workers,
        pin_memory=True,
    )

    if is_main():
        print("[KAGGLE] start predicting on test set ...")

    ids_all, probs_all = predict_for_submission(
        model, kaggle_dl, device, use_tta=args.tta
    )

    if is_main():
        pred_map = {img_id: prob for img_id, prob in zip(ids_all, probs_all)}

        sample_df["label"] = sample_df["id"].map(pred_map).astype(float)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sub_name = args.sub_name
        if not sub_name.endswith(".csv"):
            sub_name = sub_name + ".csv"
        sub_path = out_dir / sub_name

        sample_df.to_csv(sub_path, index=False)
        print(f"[KAGGLE] submission saved to {sub_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="hcd 或 hcd_macenko_eda_v2")
    ap.add_argument(
        "--labels_csv",
        type=str,
        default="hcd/train_labels.csv",
        help="Kaggle train_labels.csv 路徑（raw & macenko 都可以用同一份）",
    )
    ap.add_argument(
        "--sample_csv",
        type=str,
        default="hcd/sample_submission.csv",
        help="Kaggle sample_submission.csv 路徑",
    )
    ap.add_argument("--out_dir", default="outputs/effv2s_kaggle_4exp")
    ap.add_argument("--model", default="tf_efficientnetv2_s")  # 1k 版本

    # per-GPU batch 64，配合 grad_accum_steps=4 (7GPU 時 global ≈ 64*7*4 = 1792)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="梯度累積步數，用來模擬更大的 global batch size",
    )

    ap.add_argument("--epochs", type=int, default=15)

    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument(
        "--lr_scale_ref",
        type=int,
        default=1792,  # 64*7*4
    )

    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument(
        "--drop_rate",
        type=float,
        default=0.2,
        help="EfficientNetV2 dropout rate",
    )
    ap.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.2,
        help="EfficientNetV2 stochastic depth rate",
    )

    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--focal_alpha", type=float, default=0.6)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.15,
        help="train_labels 中拿多少比例當 validation（0~0.5），預設 0.15",
    )

    ap.add_argument(
        "--pos_weight_scale",
        type=float,
        default=1.5,
        help="放大 positive class 權重的倍率；1.0 表示不放大",
    )

    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="暫時不支援，之後如果要我可以幫你加",
    )
    ap.add_argument(
        "--tta",
        action="store_true",
        help="在 val/submission 時使用幾何 TTA（建議先不要開）",
    )

    ap.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="輸入影像邊長（H=W），預設 256；512 可能會 OOM，要自己斟酌 batch_size",
    )

    ap.add_argument(
        "--sub_name",
        type=str,
        default="submission_effv2s",
        help="輸出到 out_dir 的 submission 檔名",
    )

    ap.add_argument(
        "--use_color_aug",
        action="store_true",
        help="是否啟用顏色增強（raw: 強；macenko: 溫和）",
    )

    args = ap.parse_args()
    main(args)
