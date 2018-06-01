#!/usr/bin/env bash
# Test the segmentation model locally

python -m segmentation.train \
    --config "segmentation/config_local.ini" \
    --log=DEBUG