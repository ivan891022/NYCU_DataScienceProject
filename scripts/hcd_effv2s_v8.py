#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HCD EfficientNetV2-S (1k) + 顏色增強 v9 + 溫和幾何增強 (DDP)

實驗軸:
    1) 有 / 無 染色標準化（改 --data_dir）
       - hcd                 -> 原始 HCD
       - hcd_macenko_eda_v2  -> Macenko stain normalization
    2) 有 / 無 顏色增強（加上 --no_color_aug 就是「無顏色增強」）

設計重點（針對 AUROC 提升）:
    - 幾何增強: 水平/垂直 flip + 小角度 Rotate + Resize
    - 顏色增強（關鍵改動）:
        * 原始 HCD: 使用 OneOf，模擬真實 stain 差異，但保持「大部分 patch 顏色合理」
            - branch A: 中度 BC + Gamma
            - branch B: 輕度 BC + Gamma + 很小 RGBShift
            - branch C: 幾乎不動顏色（identity-like）
        * Macenko: 資料已標準化，顏色增強更溫和（只動亮度/對比 + 很小 gamma）
    - BCEWithLogitsLoss + pos_weight_scale（處理 positive 稀少，穩定 decision boundary）
    - Cosine LR + AMP + best checkpoint (by val AUROC) 存檔
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


def build_train_transform(
    img_size: int,
    use_color_aug: bool,
    is_macenko: bool,
):
    """建立 train augmentation（幾何 + 顏色）"""

    # 幾何增強：穩定但有 basic invariance
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

    # 顏色增強 for 原始 HCD：模擬 stain 差異，但不要炸掉顏色
    # 使用 OneOf，三個 branch:
    #   A: BC ±12% + Gamma(85~115)
    #   B: BC ±8% + Gamma(90~110) + 小 RGBShift
    #   C: 幾乎 identity（只很小 BC）
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
        p=0.9,  # 90% 機率做顏色增強，10% 保持原色
    )

    # 顏色增強 for Macenko：已標準化，只做輕微亮度/對比 + 小 gamma
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
        p=0.8,  # Macenko 顏色增強更保守
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
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        train: bool,
        img_size: int,
        use_color_aug: bool = True,
        is_macenko: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train
        self.img_size = img_size
        self.use_color_aug = use_color_aug
        self.is_macenko = is_macenko

        if train:
            self.tf = build_train_transform(
                img_size=self.img_size,
                use_color_aug=self.use_color_aug,
                is_macenko=self.is_macenko,
            )
        else:
            # val / test：只做 Resize + Normalize
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
    # 建議實驗都不要開 --tta（這裡仍然保留選項）
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

    is_macenko = "macenko" in str(args.data_dir).lower()

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

    train_set = HCDDataset(
        tr,
        img_dir,
        train=True,
        img_size=args.img_size,
        use_color_aug=(not args.no_color_aug),
        is_macenko=is_macenko,
    )
    val_set = HCDDataset(
        val,
        img_dir,
        train=False,
        img_size=args.img_size,
        use_color_aug=False,
        is_macenko=is_macenko,
    )
    test_set = HCDDataset(
        test,
        img_dir,
        train=False,
        img_size=args.img_size,
        use_color_aug=False,
        is_macenko=is_macenko,
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

    model = build_model(
        name=args.model,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

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

    # ----- loss -----
    if args.use_focal:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        if is_main():
            print("[INFO] Using FocalLoss")
    else:
        base_pw = class_pos_weight(tr["label"].values)
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
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if is_main():
        print(
            f"[INFO] world={world}  perGPU-bsz={args.batch_size}  "
            f"grad_accum={effective_accum}  "
            f"effective-global-bsz={global_bsz}  lr={base_lr:.2e}"
        )
        print(
            f"[INFO] loss={'Focal' if args.use_focal else 'BCEw'}"
        )
        print(f"[INFO] TTA for eval: {args.tta} (建議關掉)")
        print(
            f"[INFO] drop_rate={args.drop_rate:.3f}  "
            f"drop_path_rate={args.drop_path_rate:.3f}"
        )
        print(
            f"[INFO] train color aug: {not args.no_color_aug}  "
            f"(data_dir={args.data_dir}, is_macenko={is_macenko})"
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
    ap.add_argument("--out_dir", default="outputs/effv2s_color_stain_v9")
    ap.add_argument("--model", default="tf_efficientnetv2_s")  # 1k 版本

    # per-GPU batch 64，配合 grad_accum_steps=4 (7GPU 時 global = 64*7*4 = 1792)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="梯度累積步數，用來模擬更大的 global batch size",
    )

    ap.add_argument("--epochs", type=int, default=12)

    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--lr_scale_ref", type=int, default=2048)  # 64*8*4 原設定

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

    ap.add_argument(
        "--pos_weight_scale",
        type=float,
        default=1.5,
        help="放大 positive class 權重的倍率；1.0 表示不放大",
    )

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
        help="在 val/test/eval_only 時使用幾何 TTA（建議都不要開）",
    )

    ap.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="輸入影像邊長（H=W），預設 256；512 可能會 OOM，要自己斟酌 batch_size",
    )

    args = ap.parse_args()
    main(args)
