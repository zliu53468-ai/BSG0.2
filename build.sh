#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# 如果有預訓練的模型文件，跳過訓練步驟
if [ -f "models/xgb_model.pkl" ] && [ -f "models/lgbm_model.pkl" ] && [ -f "models/hmm_model.pkl" ]; then
    echo "使用預訓練模型，跳過訓練步驟..."
else
    echo "訓練模型中..."
    python train.py
fi

echo "構建成功完成！"
