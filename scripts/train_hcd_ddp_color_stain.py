#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HCD EfficientNetV2-S + 顏色增強 + 幾何增強 (DDP)
- 用於比較：原始 hcd vs 離線 Macenko (hcd_macenko_multi_eda / hcd_macenko_eda)
- 只要改 --data_dir 就能做「有 / 無染色標準化」實驗
- 只要加上 --no_color_aug 就能關掉訓練時的顏色增強（只留幾何）
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
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # TF32 可開啟加速 (Amp + CNN 很常用)
    if hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda, "matmul"
    ):
        torch.backends.cuda.matmul.allow_tf32 = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        loss = (
            (self.alpha * targets + (1 - self.alpha) * (1 - targets))
            * (1 - pt) ** self.gamma
            * bce
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def init_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def gather_lists_py(obj_list):
    """gather python list from all ranks"""
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


class HCDDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir: Path,
        train: bool,
        img_size: int,
        use_color_aug: bool = True,  # ★ 新增：是否使用顏色增強
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train
        self.img_size = img_size
        self.use_color_aug = use_color_aug

        if train:
            # ---- 幾何增強（四種設定都要一致） ----
            geo_transforms = [
                # 幾何：flip + 旋轉 + 輕微縮放/平移
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # ShiftScaleRotate 仍可用（官方建議換 Affine，但實務上 OK）
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                # 視野變化：隨機裁一點再 resize 回來
                # Albumentations 2.x 使用 size，而不是 height/width
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.85, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=cv2.INTER_LINEAR,
                    p=0.5,
                ),
                # ★ 無論 RandomResizedCrop 有沒有被套用，都再保險 Resize 一次 ★
                A.Resize(self.img_size, self.img_size),
                # 模擬掃描器 / focus 差異
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5)),
                        A.MotionBlur(blur_limit=3),
                    ],
                    p=0.15,
                ),
                A.GaussNoise(p=0.15),
            ]

            # ---- 顏色增強 block（可選） ----
            color_block = [
                # 顏色組：Brightness/Contrast 或 Hue/Sat/Value（二擇一）
                # 故意開得中等偏強一點，讓「顏色穩定性」對 Macenko 有利
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.18,
                            contrast_limit=0.18,
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=0.015,
                            sat_shift_limit=0.18,
                            val_shift_limit=0.18,
                        ),
                    ],
                    p=0.9,
                ),
                # 很輕微的 channel 抖動，打散 RGB shortcut
                A.RGBShift(
                    r_shift_limit=4,
                    g_shift_limit=4,
                    b_shift_limit=4,
                    p=0.25,
                ),
            ]

            tf_list = geo_transforms.copy()
            if use_color_aug:
                tf_list.extend(color_block)

            tf_list.extend(
                [
                    A.Normalize(
                        mean=(0.67, 0.45, 0.69),
                        std=(0.23, 0.21, 0.22),
                    ),
                    AP.ToTensorV2(),
                ]
            )

            self.tf = A.Compose(tf_list)

        else:
            # val / test：只做 Resize + Normalize（顏色保持乾淨，方便比較）
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img_path = find_img(self.img_dir, r["id"])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)  # uint8 HWC

        label = int(r["label"])
        x = self.tf(image=img)["image"]
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


# ----------------- model / metric -----------------
def build_model(
    name: str = "tf_efficientnetv2_s_in21k",
    drop_rate: float = 0.3,
    drop_path_rate: float = 0.1,
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


def class_pos_weight(labels):
    n = len(labels)
    pos = labels.sum()
    neg = n - pos
    return float(neg / max(pos, 1))


def best_thr_metrics(y_true, y_prob):
    """回傳：best_thr, best_f1, precision_at_best_thr, recall_at_best_thr"""
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
    """
    幾何 TTA（不動顏色）：
      x, flip(h/v), rot 90/180/270, rot90 + flips
    """
    if not use_tta:
        return torch.sigmoid(model(x))

    views = []
    views.append(x)
    views.append(torch.flip(x, dims=[3]))  # h-flip
    views.append(torch.flip(x, dims=[2]))  # v-flip

    x90 = torch.rot90(x, k=1, dims=[2, 3])
    x180 = torch.rot90(x, k=2, dims=[2, 3])
    x270 = torch.rot90(x, k=3, dims=[2, 3])
    views.extend([x90, x180, x270])

    views.append(torch.flip(x90, dims=[3]))  # 90 + h-flip
    views.append(torch.flip(x90, dims=[2]))  # 90 + v-flip

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

    if dist.is_initialized():
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

    # split 一律用原始 hcd 下的 csv，除非你手動指定 --split_dir
    split_dir = (
        Path(args.split_dir)
        if args.split_dir
        else Path("hcd") / "splits/seed42_70_15_15"
    )
    tr = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")
    for df in (tr, val, test):
        df.columns = [c.strip().lower() for c in df.columns]

    # datasets / loaders
    train_set = HCDDataset(
        tr,
        img_dir,
        train=True,
        img_size=args.img_size,
        use_color_aug=(not args.no_color_aug),  # ★ 這裡決定有沒有顏色增強
    )
    val_set = HCDDataset(val, img_dir, train=False, img_size=args.img_size)
    test_set = HCDDataset(test, img_dir, train=False, img_size=args.img_size)

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

    # model
    model = build_model(
        name=args.model,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    # CNN + channels_last 對效能 & 記憶體都蠻友善
    model = model.to(memory_format=torch.channels_last)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # eval-only
    if args.eval_only:
        assert args.ckpt is not None, "--eval_only 模式需要指定 --ckpt"
        if is_main():
            print(f"[INFO] Eval only mode. Loading ckpt from {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        model.module.load_state_dict(state["model"])
        au, f1, thr, prec, rec = evaluate(
            model, test_dl, device, use_tta=args.tta
        )
        if is_main():
            print(
                f"[TEST(best-ckpt)] AUROC={au:.4f}  F1*={f1:.4f}  "
                f"thr={thr:.3f}  P={prec:.4f}  R={rec:.4f}"
            )
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # loss
    if args.use_focal:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        pos_w = torch.tensor(class_pos_weight(tr["label"].values), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # LR 線性縮放（包含梯度累積）
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
        print(
            f"[INFO] pos_weight={class_pos_weight(tr['label'].values):.3f}  "
            f"loss={'Focal' if args.use_focal else 'BCEw'}"
        )
        print(f"[INFO] TTA for eval: {args.tta}")
        print(
            f"[INFO] drop_rate={args.drop_rate:.3f}  "
            f"drop_path_rate={args.drop_path_rate:.3f}"
        )
        print(
            f"[INFO] train color aug: {not args.no_color_aug}  "
            f"(data_dir={args.data_dir})"
        )

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
                # 梯度累積要把 loss 除以步數
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
                # 顯示未除之前的 loss（比較直觀）
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

    # ---------- 最終在 test 上評估 ----------
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
                f"[TEST(best)] AUROC={au:.4f}  F1*={f1:.4f}  thr={thr:.3f}  "
                f"P={prec:.4f}  R={rec:.4f} (loaded from epoch {state.get('epoch', -1)})"
            )
            print(
                f"[BEST(val)] epoch={best['epoch']}  "
                f"AUROC={best['auroc']:.4f}  F1*={best['f1']:.4f}  thr={best['thr']:.3f}"
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
    ap.add_argument("--split_dir", default="")
    ap.add_argument("--out_dir", default="outputs/effv2s_color_stain")
    ap.add_argument("--model", default="tf_efficientnetv2_s_in21k")

    # per-GPU batch，建議用 64，搭配 grad_accum_steps=4
    ap.add_argument("--batch_size", type=int, default=64)  # per GPU
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="梯度累積步數，用來模擬更大的 global batch size",
    )
    ap.add_argument("--epochs", type=int, default=40)

    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--base_lr", type=float, default=4e-4)
    ap.add_argument("--lr_scale_ref", type=int, default=1792)  # 64*7*4

    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument(
        "--drop_rate",
        type=float,
        default=0.3,
        help="EfficientNetV2 dropout rate",
    )
    ap.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        help="EfficientNetV2 stochastic depth rate",
    )

    ap.add_argument("--workers", type=int, default=8)

    # ★ 新增：關閉顏色增強
    ap.add_argument(
        "--no_color_aug",
        action="store_true",
        help="關閉訓練時的顏色增強，只保留幾何增強",
    )

    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--focal_alpha", type=float, default=0.6)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # eval-only / TTA
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="只載入 ckpt 在 test split 上做評估（不訓練）",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="eval_only 模式要載入的 checkpoint 路徑",
    )
    ap.add_argument(
        "--tta",
        action="store_true",
        help="在 val/test/eval_only 時使用幾何 TTA",
    )

    ap.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="輸入影像邊長（H=W），預設 256；如果你的 patch 不是 256 就改這裡",
    )

    args = ap.parse_args()
    main(args)
