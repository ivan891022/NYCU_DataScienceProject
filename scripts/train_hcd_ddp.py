#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, math, random, argparse
from pathlib import Path

import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm

import torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score

import albumentations as A
import albumentations.pytorch as AP
import timm

# --- optional: stain normalization (Macenko) ---
try:
    import staintools
except ImportError:
    staintools = None


# --------- utils ---------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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
    """gather python lists from all ranks"""
    world = dist.get_world_size()
    bufs = [None for _ in range(world)]
    dist.all_gather_object(bufs, obj_list)
    out = []
    for b in bufs:
        out.extend(b)
    return out


# --------- data / stain utils ---------
IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


def find_img(root: Path, img_id: str) -> Path:
    for ext in IMG_EXTS:
        p = root / f"{img_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{img_id} image not found")


def build_macenko_normalizer(img_dir: Path, df: pd.DataFrame, ref_id: str = None):
    """
    建一個 Macenko stain normalizer：
    - 若 ref_id 為 None，優先用第一張 positive patch 做 template
      （如果沒有就用第一筆資料）
    """
    if staintools is None:
        raise ImportError(
            "staintools is required for Macenko stain normalization. "
            "Please install it via `pip install staintools opencv-python`."
        )

    if ref_id is None:
        if "label" in df.columns and (df["label"] == 1).any():
            ref_id = df.loc[df["label"] == 1, "id"].iloc[0]
        else:
            ref_id = df["id"].iloc[0]

    ref_path = find_img(img_dir, ref_id)
    target = staintools.read_image(str(ref_path))  # uint8 RGB
    normalizer = staintools.StainNormalizer(method="macenko")
    normalizer.fit(target)

    def _apply(img: np.ndarray) -> np.ndarray:
        """
        img: uint8 RGB (H, W, 3)
        回傳：Macenko normalized 的 uint8 RGB
        遇到幾乎全白或 Macenko 失敗的 patch，就直接回傳原圖避免炸掉。
        """
        # 幾乎全白 / 幾乎沒有組織：沒必要做 Macenko
        if img.mean() > 240 or img.std() < 5:
            return img

        try:
            out = normalizer.transform(img)

            # 若 transform 後出現 NaN / Inf，一律丟棄
            if not np.isfinite(out).all():
                return img

            # 保證回傳 uint8
            out = np.clip(out, 0, 255).astype(np.uint8)
            return out

        except np.linalg.LinAlgError:
            # covariance / eigen decomposition 失敗 → fallback 原圖
            return img
        except Exception:
            # 任何其他奇怪錯誤也不要讓訓練炸掉
            return img

    return _apply


class HCDDataset(Dataset):
    def __init__(self, df, img_dir: Path, train=True, stain_norm=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.stain_norm = stain_norm  # 可為 None 或 callable

        if train:
            # 幾何 + 更兇的增強（在染色標準化之後）
            self.tf = A.Compose(
                [
                    # 幾何：翻轉 + 旋轉
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.10,
                        rotate_limit=15,
                        border_mode=0,
                        value=(0, 0, 0),
                        p=0.7,
                    ),

                    # 模糊 / 噪聲（輕微）
                    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),

                    # 顏色：ColorJitter + RGBShift
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                        p=0.7,
                    ),
                    A.RGBShift(
                        r_shift_limit=10,
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=0.3,
                    ),

                    # 正規化（與 HCD/PCam 常用 mean/std 接近）
                    A.Normalize(mean=(0.67, 0.45, 0.69), std=(0.23, 0.21, 0.22)),
                    AP.ToTensorV2(),
                ]
            )
        else:
            # 驗證 / 測試：只做染色標準化 + Normalize，不做隨機增強
            self.tf = A.Compose(
                [
                    A.Normalize(mean=(0.67, 0.45, 0.69), std=(0.23, 0.21, 0.22)),
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

        # --- 先做染色標準化，再丟進 Albumentations ---
        if self.stain_norm is not None:
            img = self.stain_norm(img)

        x = self.tf(image=img)["image"]
        y = torch.tensor([r["label"]], dtype=torch.float32)
        return x, y


# --------- model / loss / eval ---------
def build_model(name="efficientnetv2_s"):
    return timm.create_model(name, pretrained=True, in_chans=3, num_classes=1)


def class_pos_weight(labels):
    n = len(labels)
    pos = labels.sum()
    neg = n - pos
    return float(neg / max(pos, 1))


def best_thr_metrics(y_true, y_prob):
    """
    回傳：best_thr, best_f1, precision_at_best_thr, recall_at_best_thr
    """
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p * r / (p + r + 1e-9)

    if len(t) == 0:
        # 退化情況：所有機率一樣，用 thr=0.5
        thr = 0.5
        y_pred = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1_best = 2 * prec * rec / (prec + rec + 1e-9)
        return thr, f1_best, prec, rec

    # thresholds 對應到 p[1:], r[1:]
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
    Test-time augmentation:
      - 原圖
      - 水平翻轉
      - 垂直翻轉
      - 旋轉 90 度
    對 logits 做平均再 sigmoid。
    """
    if not use_tta:
        return torch.sigmoid(model(x))

    logits_list = []

    # 1. 原圖
    logits_list.append(model(x))

    # 2. 水平翻轉
    logits_list.append(model(torch.flip(x, dims=[3])))

    # 3. 垂直翻轉
    logits_list.append(model(torch.flip(x, dims=[2])))

    # 4. 旋轉 90°
    logits_list.append(model(torch.rot90(x, k=1, dims=[2, 3])))

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


# --------- main train / eval loop ---------
def main(args):
    seed_all(args.seed)
    local_rank, rank, world = init_ddp()
    device = torch.device(f"cuda:{local_rank}")

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "train"

    # 讀 splits
    split_dir = Path(args.split_dir) if args.split_dir else data_dir / "splits/seed42_70_15_15"
    tr = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")
    for df in (tr, val, test):
        df.columns = [c.strip().lower() for c in df.columns]

    # --- 建立 stain normalizer（若需要） ---
    stain_norm = None
    if args.stain_norm.lower() == "macenko":
        if is_main():
            print("[INFO] Building Macenko stain normalizer...")
        stain_norm = build_macenko_normalizer(img_dir, tr, args.stain_ref_id)
        if is_main():
            print("[INFO] Macenko stain normalizer ready.")

    # datasets / dataloaders
    train_set = HCDDataset(tr, img_dir, train=True, stain_norm=stain_norm)
    val_set = HCDDataset(val, img_dir, train=False, stain_norm=stain_norm)
    test_set = HCDDataset(test, img_dir, train=False, stain_norm=stain_norm)

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
    model = build_model(args.model).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # eval-only 模式（載入權重只算 test）
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
        return

    # 損失函數
    if args.use_focal:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        pos_w = torch.tensor(class_pos_weight(tr["label"].values), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # LR 按「全域 batch」線性縮放建議值
    global_bsz = args.batch_size * world
    base_lr = args.lr
    if args.lr_scale_ref:
        base_lr = args.base_lr * (global_bsz / args.lr_scale_ref)

    optim = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if is_main():
        print(
            f"[INFO] world={world}  perGPU-bsz={args.batch_size}  "
            f"global-bsz={global_bsz}  lr={base_lr:.2e}"
        )
        print(
            f"[INFO] pos_weight={class_pos_weight(tr['label'].values):.3f}  "
            f"loss={'Focal' if args.use_focal else 'BCEw'}"
        )
        print(f"[INFO] stain_norm={args.stain_norm}")
        print(f"[INFO] TTA for eval: {args.tta}")

    best = {"auroc": -1.0, "f1": -1.0, "thr": 0.5, "epoch": -1}
    ckpt_path = Path(args.out_dir) / f"{args.model}_best.pth"

    # --------- training loop ----------
    for ep in range(1, args.epochs + 1):
        model.train()
        train_samp.set_epoch(ep)
        pbar = tqdm(train_dl, disable=not is_main(), desc=f"Epoch {ep}/{args.epochs}")

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if is_main():
                pbar.set_postfix(loss=float(loss.detach().mean()))

        sched.step()

        # 每個 epoch 在 val 上看 AUROC / F1 / Precision / Recall（閾值自動找）
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

    # --------- 用 best ckpt 在 test 上做最終評估 ----------
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split_dir", default="")
    ap.add_argument("--out_dir", default="outputs/effv2s_ddp")
    ap.add_argument("--model", default="efficientnetv2_s")
    ap.add_argument("--batch_size", type=int, default=512)  # 每顆 GPU 的 batch
    ap.add_argument("--epochs", type=int, default=20)

    ap.add_argument("--lr", type=float, default=3e-4)       # 若不使用 lr_scale_ref 就直接用這個
    ap.add_argument("--base_lr", type=float, default=3e-4)  # 參考學習率（對應 lr_scale_ref 的全域 batch）
    ap.add_argument("--lr_scale_ref", type=int, default=2048)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--focal_alpha", type=float, default=0.6)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # ---- stain normalization args ----
    ap.add_argument(
        "--stain_norm",
        type=str,
        default="macenko",
        choices=["none", "macenko"],
        help="選擇染色標準化方法；'none' 表示不做",
    )
    ap.add_argument(
        "--stain_ref_id",
        type=str,
        default=None,
        help="當使用 Macenko 時，可指定一張 image id 當作 reference patch。預設自動選第一張 positive。",
    )

    # ---- eval-only args ----
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

    # ---- TTA flag ----
    ap.add_argument(
        "--tta",
        action="store_true",
        help="在 val/test/eval_only 時使用 flip+rotate TTA",
    )

    args = ap.parse_args()
    main(args)
