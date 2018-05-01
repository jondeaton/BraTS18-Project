#!/usr/bin/env bash
# Test the segmentation model locally

timestamp=`date +%s`

job_name="signs_local_$timestamp"
tensorboard_dir="SIGNS/tensorboard"
dataset_dir="SIGNS/datasets"

python -m segmentation.train \
    --job-name "$job_name" \
    --tensorboard "$tensorboard_dir" \
    --dataset-dir "$dataset_dir" \
    --learning-rate "0.0001" \
    --epochs 1500 \
    --mini-batch 128 \
    --log=DEBUG