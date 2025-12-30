#!/bin/bash
conda activate mmdet3

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
NGPU=7

# 1) raw：無顏色增強、無染色標準化
torchrun --nnodes=1 --nproc_per_node=$NGPU scripts/hcd_effv2s_kaggle_4exp.py \
  --data_dir hcd \
  --labels_csv hcd/train_labels.csv \
  --sample_csv hcd/sample_submission.csv \
  --out_dir outputs/effv2s_raw_nocolor \
  --sub_name submission_effv2s_raw_nocolor.csv \
  --batch_size 64 \
  --epochs 15 \
  --amp

# 2) raw + 顏色增強
torchrun --nnodes=1 --nproc_per_node=$NGPU scripts/hcd_effv2s_kaggle_4exp.py \
  --data_dir hcd \
  --labels_csv hcd/train_labels.csv \
  --sample_csv hcd/sample_submission.csv \
  --out_dir outputs/effv2s_raw_color \
  --sub_name submission_effv2s_raw_color.csv \
  --batch_size 64 \
  --epochs 15 \
  --amp \
  --use_color_aug

# 3) Macenko：染色標準化、無顏色增強
torchrun --nnodes=1 --nproc_per_node=$NGPU scripts/hcd_effv2s_kaggle_4exp.py \
  --data_dir hcd_macenko_eda_v2 \
  --labels_csv hcd/train_labels.csv \
  --sample_csv hcd/sample_submission.csv \
  --out_dir outputs/effv2s_macenko_nocolor \
  --sub_name submission_effv2s_macenko_nocolor.csv \
  --batch_size 64 \
  --epochs 15 \
  --amp

# 4) Macenko + 顏色增強
torchrun --nnodes=1 --nproc_per_node=$NGPU scripts/hcd_effv2s_kaggle_4exp.py \
  --data_dir hcd_macenko_eda_v2 \
  --labels_csv hcd/train_labels.csv \
  --sample_csv hcd/sample_submission.csv \
  --out_dir outputs/effv2s_macenko_color \
  --sub_name submission_effv2s_macenko_color.csv \
  --batch_size 64 \
  --epochs 15 \
  --amp \
  --use_color_aug
