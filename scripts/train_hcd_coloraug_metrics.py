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

# --------- utils ---------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p*targets + (1-p)*(1-targets)
        loss = ((self.alpha*targets + (1-self.alpha)*(1-targets)) * (1-pt)**self.gamma * bce)
        return loss.mean() if self.reduction=="mean" else loss.sum()

def init_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def is_main():
    return (not dist.is_initialized()) or dist.get_rank()==0

def gather_lists_py(obj_list):
    world = dist.get_world_size()
    bufs = [None for _ in range(world)]
    dist.all_gather_object(bufs, obj_list)
    out = []
    for b in bufs: out.extend(b)
    return out

# --------- data ---------
IMG_EXTS = [".tif",".tiff",".png",".jpg",".jpeg"]
def find_img(root: Path, img_id: str) -> Path:
    for ext in IMG_EXTS:
        p = root/f"{img_id}{ext}"
        if p.exists(): return p
    raise FileNotFoundError(f"{img_id} image not found in {root}")

class HCDDataset(Dataset):
    """
    加入「溫和但多樣」的色彩增強（依前面討論）：
    - 幾何：水平/垂直翻轉、隨機 90 度旋轉
    - 顏色：在 ColorJitter、HSV 變換、RGBShift 三者中擇一（p=0.7）
    - 少量噪聲/對比度處理：Gamma or GaussNoise（很小機率）
    - 統一 Normalize（沿用你之前的 mean/std）
    """
    def __init__(self, df, img_dir: Path, train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        if train:
            self.tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                A.OneOf([
                    A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.03, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=8, p=1.0),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                ], p=0.7),

                A.OneOf([
                    A.RandomGamma(gamma_limit=(85,115), p=1.0),
                    A.GaussNoise(var_limit=(5.0, 15.0), p=1.0),
                    A.NoOp(p=1.0)
                ], p=0.15),

                A.Normalize(mean=(0.67,0.45,0.69), std=(0.23,0.21,0.22)),
                AP.ToTensorV2()
            ])
        else:
            self.tf = A.Compose([
                A.Normalize(mean=(0.67,0.45,0.69), std=(0.23,0.21,0.22)),
                AP.ToTensorV2()
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(find_img(self.img_dir, r["id"])).convert("RGB")
        x = self.tf(image=np.array(img))["image"]
        y = torch.tensor([r["label"]], dtype=torch.float32)
        return x, y

# --------- model / loss / eval ---------
def build_model(name="efficientnetv2_s"):
    # timm 某些環境下 'efficientnetv2_s' 沒預訓練權重，改用 tf_* 對應
    try:
        return timm.create_model(name, pretrained=True, in_chans=3, num_classes=1)
    except RuntimeError as e:
        if "No pretrained weights exist" in str(e):
            return timm.create_model("tf_efficientnetv2_s", pretrained=True, in_chans=3, num_classes=1)
        raise

def class_pos_weight(labels):
    n=len(labels); pos=labels.sum(); neg=n-pos
    return float(neg/max(pos,1))

def best_thr_f1(y_true, y_prob):
    p,r,t = precision_recall_curve(y_true, y_prob)
    f1 = 2*p*r/(p+r+1e-9)
    idx = int(np.nanargmax(f1))
    thr = float(t[idx]) if idx < len(t) else 0.5
    return thr, float(f1[idx])

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    probs_all, targs_all = [], []
    for x,y in dl:
        x = x.to(device, non_blocking=True)
        pr = torch.sigmoid(model(x)).squeeze(1).cpu().numpy().tolist()
        tg = y.squeeze(1).cpu().numpy().tolist()
        probs_all.extend(pr); targs_all.extend(tg)

    if dist.is_initialized():
        probs_all = gather_lists_py(probs_all)
        targs_all = gather_lists_py(targs_all)

    y_prob = np.array(probs_all); y_true = np.array(targs_all, dtype=int)
    auroc = roc_auc_score(y_true, y_prob)
    thr, f1 = best_thr_f1(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    return auroc, f1, prec, rec, thr

# --------- main train loop ---------
def main(args):
    seed_all(args.seed)
    local_rank, rank, world = init_ddp()
    device = torch.device(f"cuda:{local_rank}")

    data_dir = Path(args.data_dir)
    img_dir = data_dir/"train"

    split_dir = Path(args.split_dir) if args.split_dir else data_dir/"splits/seed42_70_15_15"
    tr = pd.read_csv(split_dir/"train.csv"); val = pd.read_csv(split_dir/"val.csv"); test = pd.read_csv(split_dir/"test.csv")
    for df in (tr,val,test): df.columns=[c.strip().lower() for c in df.columns]

    train_set = HCDDataset(tr, img_dir, train=True)
    val_set   = HCDDataset(val, img_dir, train=False)
    test_set  = HCDDataset(test, img_dir, train=False)

    train_samp = DistributedSampler(train_set, shuffle=True, drop_last=False)
    val_samp   = DistributedSampler(val_set, shuffle=False, drop_last=False)
    test_samp  = DistributedSampler(test_set, shuffle=False, drop_last=False)

    train_dl = DataLoader(train_set, batch_size=args.batch_size, sampler=train_samp,
                          num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers>0))
    val_dl   = DataLoader(val_set,   batch_size=args.batch_size, sampler=val_samp,
                          num_workers=args.workers, pin_memory=True)
    test_dl  = DataLoader(test_set,  batch_size=args.batch_size, sampler=test_samp,
                          num_workers=args.workers, pin_memory=True)

    model = build_model(args.model).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if args.use_focal:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        pos_w = torch.tensor(class_pos_weight(tr["label"].values), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    global_bsz = args.batch_size * world
    base_lr = args.lr
    if args.lr_scale_ref:
        base_lr = args.base_lr * (global_bsz / args.lr_scale_ref)
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if is_main():
        print(f"[INFO] world={world}  perGPU-bsz={args.batch_size}  global-bsz={global_bsz}  lr={base_lr:.2e}")
        print(f"[INFO] pos_weight={class_pos_weight(tr['label'].values):.3f}  loss={'Focal' if args.use_focal else 'BCEw'}")

    best = {"auroc":-1}
    for ep in range(1, args.epochs+1):
        model.train()
        train_samp.set_epoch(ep)
        pbar = tqdm(train_dl, disable=not is_main(), desc=f"Epoch {ep}/{args.epochs}")
        for x,y in pbar:
            x=x.to(device, non_blocking=True); y=y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            if is_main(): pbar.set_postfix(loss=float(loss.detach().mean()))
        sched.step()

        au, f1, pr, rc, thr = evaluate(model, val_dl, device)
        if is_main():
            print(f"  -> val AUROC={au:.4f}  F1*={f1:.4f}  P={pr:.4f}  R={rc:.4f}  thr={thr:.3f}")
        if au > best["auroc"]:
            best.update({"auroc":au, "f1":f1, "P":pr, "R":rc, "thr":thr, "epoch":ep})
            if is_main():
                out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model":model.module.state_dict(), **best},
                           out_dir/f"{args.model}_best.pth")

    au, f1, pr, rc, thr = evaluate(model, test_dl, device)
    if is_main():
        print(f"[TEST] AUROC={au:.4f}  F1*={f1:.4f}  P={pr:.4f}  R={rc:.4f}  thr={thr:.3f}")
        print(f"[BEST(val)] epoch={best['epoch']} AUROC={best['auroc']:.4f} F1*={best['f1']:.4f} "
              f"P={best['P']:.4f} R={best['R']:.4f} thr={best['thr']:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split_dir", default="")
    ap.add_argument("--out_dir", default="outputs/effv2s_ddp_color")
    ap.add_argument("--model", default="efficientnetv2_s")     # 會自動 fallback 到 tf_efficientnetv2_s
    ap.add_argument("--batch_size", type=int, default=512)     # 每顆 GPU 的 batch
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--lr_scale_ref", type=int, default=2048)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--focal_alpha", type=float, default=0.6)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); main(args)
