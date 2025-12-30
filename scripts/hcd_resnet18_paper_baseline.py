#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HCD – ResNet18 (paper ResNet18all-style baseline) on your 70/15/15 splits

Paper: "ResNet for Histopathologic Cancer Detection, the Deeper, the Better?"

Main setting we mimic:
    - Backbone: ResNet-18 pretrained on ImageNet
    - Fine-tune ALL layers (ResNet18all style)
    - Input: H&E patches (96x96), random crop to 90x90 (train)
    - Geometric aug: random H/V flip
    - No color augmentation, ImageNet mean/std
    - Your stratified 70/15/15 splits:
        hcd/splits/seed42_70_15_15/{train,val,test}.csv

Also:
    - After training, evaluate on test split
    - Use best-val checkpoint to predict Kaggle test/ and write submission csv
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

import albumentations as A
import albumentations.pytorch as AP
import timm

from torch.cuda.amp import GradScaler, autocast


IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

# =========================================================
# utils
# =========================================================

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
    """Assume launch by torchrun."""
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


# =========================================================
# data / aug
# =========================================================

def find_img(root: Path, img_id: str) -> Path:
    for ext in IMG_EXTS:
        p = root / f"{img_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{img_id} image not found under {root}")


def build_train_transform(crop_size: int):
    """
    Paper-style augment:
        - random crop to 90x90
        - random horiz/vert flip
        - channel-wise normalize (ImageNet mean/std)
    """
    tf_list = [
        A.RandomCrop(crop_size, crop_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        AP.ToTensorV2(),
    ]
    return A.Compose(tf_list)


def build_eval_transform(crop_size: int):
    """
    Val/Test: deterministic center crop + normalize.
    """
    tf_list = [
        A.CenterCrop(crop_size, crop_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        AP.ToTensorV2(),
    ]
    return A.Compose(tf_list)


class HCDDataset(Dataset):
    """有 label 的 train / val / test"""

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        train: bool,
        crop_size: int,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train
        self.crop_size = crop_size

        if train:
            self.tf = build_train_transform(crop_size=self.crop_size)
        else:
            self.tf = build_eval_transform(crop_size=self.crop_size)

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
        crop_size: int,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.crop_size = crop_size
        self.tf = build_eval_transform(crop_size=self.crop_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img_id = self.df.loc[i, "id"]
        img_path = find_img(self.img_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        x = self.tf(image=img)["image"]
        return x, img_id


# =========================================================
# model / metric
# =========================================================

def build_model(
    name: str = "resnet18",  # ResNet-18, ImageNet-1k
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    """
    Paper: ResNet18 pre-trained on ImageNet, fine-tune all layers.
    Output: 1 logit (BCE)  => sigmoid -> prob of cancer.
    """
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
    # 正常建議 TTA 關掉；這裡保留選項
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


# =========================================================
# main train loop
# =========================================================

def main(args):
    seed_all(args.seed)
    local_rank, rank, world = init_ddp()
    device = torch.device(f"cuda:{local_rank}")

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "train"

    # ---- read your 70/15/15 splits ----
    split_dir = (
        Path(args.split_dir)
        if args.split_dir
        else data_dir / "splits/seed42_70_15_15"
    )
    tr = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    te = pd.read_csv(split_dir / "test.csv")

    for df in (tr, val, te):
        df.columns = [c.strip().lower() for c in df.columns]

    if is_main():
        pos_tr = tr["label"].mean()
        pos_val = val["label"].mean()
        pos_te = te["label"].mean()
        print(
            f"[DATA] train={len(tr)} (pos={pos_tr:.4f})  "
            f"val={len(val)} (pos={pos_val:.4f})  "
            f"test={len(te)} (pos={pos_te:.4f})"
        )
        print(f"[DATA] data_dir={data_dir}, splits={split_dir}")
        print(f"[DATA] crop_size={args.crop_size} (paper: 90x90)")

    # Dataset / Dataloader
    train_set = HCDDataset(
        tr,
        img_dir,
        train=True,
        crop_size=args.crop_size,
    )
    val_set = HCDDataset(
        val,
        img_dir,
        train=False,
        crop_size=args.crop_size,
    )
    test_set = HCDDataset(
        te,
        img_dir,
        train=False,
        crop_size=args.crop_size,
    )

    train_samp = DistributedSampler(train_set, shuffle=True, drop_last=False)
    val_samp = DistributedSampler(val_set, shuffle=False, drop_last=False)
    test_samp = DistributedSampler(test_set, shuffle=False, drop_last=False)

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
    test_dl = DataLoader(
        test_set,
        batch_size=args.batch_size,
        sampler=test_samp,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Model: ResNet18, fine-tune ALL layers
    model = build_model(
        name=args.model,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    model = model.to(memory_format=torch.channels_last)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if args.eval_only:
        raise NotImplementedError(
            "--eval_only 還沒實作，需要可以再加"
        )

    # ----- loss -----
    if args.use_focal:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        if is_main():
            print("[INFO] Using FocalLoss")
    else:
        base_pw = class_pos_weight(tr["label"].values)
        pw = base_pw * args.pos_weight_scale
        pos_w = torch.tensor(pw, device=device)
        # BCEWithLogitsLoss on single logit
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        if is_main():
            print(
                f"[INFO] Using BCEWithLogitsLoss, base_pos_weight={base_pw:.3f}, "
                f"scale={args.pos_weight_scale:.3f}, final_pos_weight={pw:.3f}"
            )

    # LR scaling (跟 EffNet script 一樣邏輯)
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
        print("[INFO] backbone=ResNet18 (ResNet18all-style)")
        print(
            f"[INFO] drop_rate={args.drop_rate:.3f}  "
            f"drop_path_rate={args.drop_path_rate:.3f}"
        )
        print(
            "[INFO] aug = random 90x90 crop + H/V flip, "
            "no color augment, ImageNet mean/std"
        )

    best = {"auroc": -1.0, "f1": -1.0, "thr": 0.5, "epoch": -1}
    ckpt_path = Path(args.out_dir) / f"{args.model}_best.pth"

    # ---------------- training loop ----------------
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

    # ---------------- reload best ckpt ----------------
    if dist.is_initialized():
        dist.barrier()

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(state["model"])
    else:
        if is_main():
            print("[WARN] best checkpoint not found, using last-epoch weights.")

    # final val + test eval
    au, f1, thr, prec, rec = evaluate(
        model, val_dl, device, use_tta=args.tta
    )
    if is_main():
        print(
            f"[VAL(final)] AUROC={au:.4f}  F1*={f1:.4f}  thr={thr:.3f}  "
            f"P={prec:.4f}  R={rec:.4f}"
        )
        if best["epoch"] > 0:
            print(
                f"[BEST(val)] epoch={best['epoch']}  "
                f"AUROC={best['auroc']:.4f}  F1*={best['f1']:.4f}  thr={best['thr']:.3f}"
            )

    au_t, f1_t, thr_t, prec_t, rec_t = evaluate(
        model, test_dl, device, use_tta=args.tta
    )
    if is_main():
        print(
            f"[TEST] AUROC={au_t:.4f}  F1*={f1_t:.4f}  thr={thr_t:.3f}  "
            f"P={prec_t:.4f}  R={rec_t:.4f}"
        )

    # ---------------- Kaggle submission ----------------
    if dist.is_initialized():
        dist.barrier()

    sample_sub_path = data_dir / "sample_submission.csv"
    sample_df = pd.read_csv(sample_sub_path)
    test_ids_df = sample_df[["id"]].copy()
    kaggle_img_dir = data_dir / "test"

    kaggle_set = HCDKaggleTestDataset(
        test_ids_df,
        kaggle_img_dir,
        crop_size=args.crop_size,
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
    ap.add_argument("--data_dir", required=True)
    ap.add_argument(
        "--split_dir",
        default="",
        help="若留空，預設使用 data_dir/splits/seed42_70_15_15",
    )
    ap.add_argument("--out_dir", default="outputs/resnet18_paper_baseline")
    ap.add_argument("--model", default="resnet18")

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
        default=0.0,   # paper 沒特別講 dropout，這裡先關閉
        help="ResNet dropout rate",
    )
    ap.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.0,
        help="stochastic depth rate (ResNet 這裡預設不用)",
    )

    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--focal_alpha", type=float, default=0.6)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--pos_weight_scale",
        type=float,
        default=1.5,
        help="放大 positive class 權重的倍率；1.0 表示不放大",
    )

    ap.add_argument(
        "--val_only",
        dest="eval_only",
        action="store_true",
        help="(暫未實作) 只做 evaluate / submission",
    )

    ap.add_argument(
        "--tta",
        action="store_true",
        help="在 val/test/submission 時使用幾何 TTA（建議先不要開）",
    )

    ap.add_argument(
        "--crop_size",
        type=int,
        default=90,   # paper: 90x90 crop
        help="訓練/測試使用的裁切大小",
    )

    ap.add_argument(
        "--sub_name",
        type=str,
        default="submission_resnet18_paper_baseline",
        help="輸出到 out_dir 的 submission 檔名",
    )

    args = ap.parse_args()
    main(args)
