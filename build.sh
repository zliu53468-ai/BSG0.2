#!/usr/bin/env bash
set -o errexit

# 强制使用 Python 3.11
PYTHON_VERSION=3.11.9

# 检查当前 Python 版本
CURRENT_PYTHON=$(python --version 2>&1 | awk '{print $2}')
if [ "$CURRENT_PYTHON" != "$PYTHON_VERSION" ]; then
    echo "错误: 需要 Python $PYTHON_VERSION，但当前是 $CURRENT_PYTHON"
    echo "请确保 runtime.txt 正确设置为 $PYTHON_VERSION"
    exit 1
fi

# 安装依赖
pip install -r requirements.txt

# 检查模型是否存在，如果不存在则训练
if [ ! -f "models/xgb_model.pkl" ] || [ ! -f "models/lgbm_model.pkl" ] || [ ! -f "models/hmm_model.pkl" ]; then
    echo "训练模型中..."
    python train.py
else
    echo "使用预训练模型..."
fi

echo "构建成功完成！"
