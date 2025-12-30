# eda_hcd_kde_raw_vs_macenko.py
import os
import random
import argparse
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None


def sample_image_paths(root_dir, n_samples=1000):
    # 把底下所有影像抓出來（副檔名你可以依實際情況再加）
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    all_paths = []
    for e in exts:
        all_paths.extend(glob(os.path.join(root_dir, "**", e), recursive=True))

    if len(all_paths) == 0:
        raise ValueError(f"No images found under {root_dir}")

    if len(all_paths) < n_samples:
        n_samples = len(all_paths)

    return random.sample(all_paths, n_samples)


def compute_channel_means(img_paths):
    means_r, means_g, means_b = [], [], []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r = img[:, :, 0].mean()
        g = img[:, :, 1].mean()
        b = img[:, :, 2].mean()
        means_r.append(r)
        means_g.append(g)
        means_b.append(b)
    return np.array(means_r), np.array(means_g), np.array(means_b)


def plot_kde(ax, r, g, b, title):
    colors = [("R", "r", r), ("G", "g", g), ("B", "b", b)]
    xs = np.linspace(0, 255, 256)

    for label, c, arr in colors:
        if gaussian_kde is not None:
            kde = gaussian_kde(arr)
            ys = kde(xs)
            ax.plot(xs, ys, label=label)
        else:
            # 沒有 scipy 就用 histogram 當近似
            ax.hist(arr, bins=50, range=(0, 255), density=True,
                    histtype="step", label=label)

    ax.set_title(title)
    ax.set_xlabel("Mean Intensity")
    ax.set_ylabel("Density")
    ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True,
                        help="Raw HCD patches 的資料夾，例如 hcd/train")
    parser.add_argument("--macenko_dir", required=True,
                        help="Macenko normalized patches 的資料夾，例如 hcd_macenko_eda_v2/train")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--out_path", default="reports/raw_vs_macenko_kde.png")
    args = parser.parse_args()

    random.seed(42)

    print("Sampling raw patches...")
    raw_paths = sample_image_paths(args.raw_dir, args.n_samples)
    raw_r, raw_g, raw_b = compute_channel_means(raw_paths)

    print("Sampling macenko patches...")
    mac_paths = sample_image_paths(args.macenko_dir, args.n_samples)
    mac_r, mac_g, mac_b = compute_channel_means(mac_paths)

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    plot_kde(ax1, raw_r, raw_g, raw_b, f"Raw (n={len(raw_r)})")

    ax2 = plt.subplot(1, 2, 2)
    plot_kde(ax2, mac_r, mac_g, mac_b, f"Macenko (n={len(mac_r)})")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=200)
    print(f"Saved figure to {args.out_path}")


if __name__ == "__main__":
    main()
