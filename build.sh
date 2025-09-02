#!/usr/bin/env bash
set -o errexit

echo "开始安装依赖..."
pip install -r requirements.txt

# 检查模型是否存在，如果不存在则使用简化训练
if [ ! -f "models/xgb_model.pkl" ] || [ ! -f "models/lgbm_model.pkl" ] || [ ! -f "models/hmm_model.pkl" ]; then
    echo "训练简化模型中..."
    python train.py --lightweight  # 添加轻量级训练选项
else
    echo "使用预训练模型..."
fi

echo "构建成功完成！"
