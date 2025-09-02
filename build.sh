#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# 如果 models 目录不存在或为空，则训练模型
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "Training models..."
    python train.py
else
    echo "Using pre-trained models..."
fi

echo "Build completed successfully!"
