#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. 安裝所有 Python 套件
pip install -r requirements.txt

# 2. 執行訓練腳本來建立模型檔案
python train.py

echo "模型訓練完成，建置成功！"
