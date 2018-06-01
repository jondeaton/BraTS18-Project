#!/usr/bin/env bash
# Script for submitting Segmentation job to Google ML Engine

project_directory="segmentation"
project_name="BraTS"

# Get the current project ID
project_id=`gcloud config list project --format "value(core.project)"`

timestamp=`date +%s`
job_name=$project_name"_job_$timestamp"
bucket_name="brats-20x"
cloud_config="$project_directory/cloudml-gpu.yaml"
train_config="config_gcp.ini"
job_dir="gs://$bucket_name/segmentation"  # where to save
module="$project_directory.train"
package="./$project_directory"
region="us-east1"
runtime="1.0"

gcloud ml-engine jobs submit training "$job_name" \
    --job-dir "$job_dir" \
    --runtime-version "$runtime" \
    --module-name "$module" \
    --package-path "$package" \
    --region "$region" \
    --config="$cloud_config" \
    -- \
    --config="$train_config" \
    --job-name "$job_name" \
    --log=DEBUG \
    --google-cloud