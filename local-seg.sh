#!/usr/bin/env bash
# Test the segmentation model locally

timestamp=`date +%s`

job_name="brats_local_$timestamp"
tensorboard_dir="segmentation/tensorboard"
brats_root="~/Datasets/BraTS"

python -m segmentation.train \
    --job-name "$job_name" \
    --tensorboard "$tensorboard_dir" \
    --brats "$brats_root" \
    --learning-rate "0.0001" \
    --epochs 1500 \
    --mini-batch 128 \
    --log=DEBUG