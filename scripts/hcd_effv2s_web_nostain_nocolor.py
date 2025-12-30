#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HCD EfficientNetV2-S (ImageNet-1k, no stain norm, no color aug) + DDP

- Dataset:
    data_dir/
        train/  (HCD train images, .tif)
        test/   (HCD test images; 你的 split 會從 train_labels 取 70/15/15)
        train_labels.csv  (id,label)
        splits/seed42_70_15_15/{train,val,test}.csv  (id,label)

- This script:
    * 用你現成的 split CSV (70/15/15) 當 train / val / test
    * EfficientNetV2-S，只做幾何增強 (flip + 小角度 rotate + resize)
    * 不做染色標準化、不做顏色增強
    * BCEWithLogitsLoss + pos_weight 處理不平衡
    * 以 val AUROC 挑 best checkpoint 存到 out_dir
    * 結束時載入 best ckpt，在 test split 上報 AUROC / F1 / P / R

建議訓練指令（7 GPUs, CUDA_VISIBLE_DEVICES=1-7）：

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=7 \
  scripts/hcd_effv2s_web_nostain_nocolor.py \
  --data_dir hcd \
  --split_dir hcd/splits/seed42_70_15_15 \
  --out_dir outputs/effv2s_web_nostain_nocolor \
  --batch_size 64 \
  --epochs 15 \
  --img_size 256 \
  --amp
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


def init_ddp():
    """Initialize DDP (torchrun style)."""
    if dist.is_initialized():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def gather_lists_py(obj_list):
    """Gather Python lists across ranks (for metrics)."""
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


def build_train_transform(img_size: int):
    """Geometry-only augmentation (no color aug, no stain norm)."""
    tf_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ),
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.67, 0.45, 0.69),
            std=(0.23, 0.21, 0.22),
        ),
        AP.ToTensorV2(),
    ]
    return A.Compose(tf_list)


class HCDDataset(Dataset):
    """Dataset for split CSVs (train/val/test)."""

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        train: bool,
        img_size: int,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train
        self.img_size = img_size

        if train:
            self.tf = build_train_transform(img_size=self.img_size)
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


# ----------------- model / metric -----------------
def build_model(
    name: str = "tf_efficientnetv2_s",  # ImageNet-1k version
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
    """Find best F1 threshold and corresponding precision/recall."""
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
def evaluate(model, dl, device, use_tta: bool = False):
    model.eval()
    probs_all, targs_all = [], []

    for x, y in dl:
        x = x.to(device, non_blocking=True)

        if use_tta:
            # simple geometry TTA (optional,預設關閉)
            views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]
            x90 = torch.rot90(x, k=1, dims=[2, 3])
            x180 = torch.rot90(x, k=2, dims=[2, 3])
            x270 = torch.rot90(x, k=3, dims=[2, 3])
            views.extend([x90, x180, x270])
            views.append(torch.flip(x90, dims=[3]))
            views.append(torch.flip(x90, dims=[2]))
            logits_list = [model(v) for v in views]
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
            pr_tensor = torch.sigmoid(logits)
        else:
            pr_tensor = torch.sigmoid(model(x))

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


# ----------------- main train loop -----------------
def main(args):
    seed_all(args.seed)
    local_rank, rank, world = init_ddp()
    device = torch.device(f"cuda:{local_rank}")

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "train"

    # ---- read splits ----
    split_dir = Path(args.split_dir)
    tr = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")
    for df in (tr, val, test):
        df.columns = [c.strip().lower() for c in df.columns]

    if is_main():
        print(
            f"[DATA] train={len(tr)} (pos={tr['label'].mean():.4f})  "
            f"val={len(val)} (pos={val['label'].mean():.4f})  "
            f"test={len(test)} (pos={test['label'].mean():.4f})"
        )
        print(f"[DATA] data_dir={data_dir}")

    # Dataset / Dataloader
    train_set = HCDDataset(
        tr,
        img_dir,
        train=True,
        img_size=args.img_size,
    )
    val_set = HCDDataset(
        val,
        img_dir,
        train=False,
        img_size=args.img_size,
    )
    test_set = HCDDataset(
        test,
        img_dir,
        train=False,
        img_size=args.img_size,
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

    # Model
    model = build_model(
        name=args.model,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    model = model.to(memory_format=torch.channels_last)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ----- loss -----
    base_pw = class_pos_weight(tr["label"].values)
    pw = base_pw * args.pos_weight_scale
    pos_w = torch.tensor(pw, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    if is_main():
        print(
            f"[INFO] Using BCEWithLogitsLoss, base_pos_weight={base_pw:.3f}, "
            f"scale={args.pos_weight_scale:.3f}, final_pos_weight={pw:.3f}"
        )

    # LR linear scaling
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
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if is_main():
        print(
            f"[INFO] world={world}  perGPU-bsz={args.batch_size}  "
            f"grad_accum={effective_accum}  "
            f"effective-global-bsz={global_bsz}  lr={base_lr:.2e}"
        )
        print(f"[INFO] loss=BCEw")
        print(f"[INFO] TTA for eval: {args.tta} (recommended off)")
        print(
            f"[INFO] drop_rate={args.drop_rate:.3f}  "
            f"drop_path_rate={args.drop_path_rate:.3f}"
        )
        print(
            "[INFO] aug: geometry-only (flip/rotate/resize), "
            "no color aug, no stain normalization"
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

            with torch.amp.autocast("cuda", enabled=args.amp):
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

    # ---------- evaluate on test with best ckpt ----------
    if dist.is_initialized():
        dist.barrier()

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(state["model"])
        au, f1, thr, prec, rec = evaluate(
            model, test_dl, device, use_tta=args.tta
        )
        if is_main():
            print(
                f"[TEST(best)] AUROC={au:.4f}  F1*={f1:.4f}  "
                f"thr={thr:.3f}  P={prec:.4f}  R={rec:.4f} "
                f"(loaded from epoch {state.get('epoch', -1)})"
            )
            print(
                f"[BEST(val)] epoch={best['epoch']}  "
                f"AUROC={best['auroc']:.4f}  F1*={best['f1']:.4f}  "
                f"thr={best['thr']:.3f}"
            )
    else:
        if is_main():
            print("[WARN] best checkpoint not found, evaluating last-epoch weights.")
        au, f1, thr, prec, rec = evaluate(
            model, test_dl, device, use_tta=args.tta
        )
        if is_main():
            print(
                f"[TEST(last)] AUROC={au:.4f}  F1*={f1:.4f}  "
                f"thr={thr:.3f}  P={prec:.4f}  R={rec:.4f}"
            )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument(
        "--split_dir",
        required=True,
        help="Path to split dir containing train.csv/val.csv/test.csv",
    )
    ap.add_argument(
        "--out_dir",
        default="outputs/effv2s_web_nostain_nocolor",
    )
    ap.add_argument("--model", default="tf_efficientnetv2_s")

    # per-GPU batch 64, grad_accum=4 -> global ~64 * world * 4
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )

    ap.add_argument("--epochs", type=int, default=15)

    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument(
        "--lr_scale_ref",
        type=int,
        default=1792,  # 64 * 7 * 4 for your 7-GPU setting
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

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--pos_weight_scale",
        type=float,
        default=1.5,
        help="Scale factor for positive class weight in BCE loss.",
    )

    ap.add_argument(
        "--tta",
        action="store_true",
        help="Use simple geometry TTA for val/test (default off).",
    )

    ap.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input image size (H=W).",
    )

    args = ap.parse_args()
    main(args)
