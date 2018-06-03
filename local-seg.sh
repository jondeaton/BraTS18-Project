#!/usr/bin/env bash
# Test the segmentation model locally

python -m segmentation.train \
    --config "config_local.ini" \
    --log=DEBUG