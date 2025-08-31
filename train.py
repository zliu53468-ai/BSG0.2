# -*- coding: utf-8 -*-
import numpy as np
import joblib
import os
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from hmmlearn import hmm

# 忽略 HMM learn 的一些舊版本警告
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# =============================================================================
# 全域設定
# =============================================================================
MODEL_DIR = 'models'
SYNTHETIC_DATA_SIZE = 5000 # 生成5000筆模擬數據以進行穩健的訓練
N_FEATURES_WINDOW = 20     # 提取特徵時回看的時間窗口大小
LABEL_MAP = {'B': 0, 'P': 1}

# =============================================================================
# 特徵提取函式 (此函式必須與 app.py 中的版本邏輯一致)
# =============================================================================
def extract_features_for_training(full_roadmap):
    """從完整的路單中為訓練提取特徵和標籤。"""
    features_list = []
    labels = []
    
    # 需要至少 N+1 個數據點才能提取第一組特徵及其標籤
    if len(full_roadmap) <= N_FEATURES_WINDOW:
        return np.array([]), np.array([])

    for i in range(N_FEATURES_WINDOW, len(full_roadmap)):
        window = full_roadmap[i-N_FEATURES_WINDOW:i]
        label = full_roadmap[i]

        # 只為有明確標籤 (B/P) 的數據點創建訓練樣本
        if label not in LABEL_MAP:
            continue

        b_count = window.count('B')
        p_count = window.count('P')
        total = b_count + p_count

        b_ratio = b_count / total if total > 0 else 0.5
        p_ratio = p_count / total if total > 0 else 0.5
        
        streak = 0
        last_result = None
        for item in reversed(window):
            if item in ['B', 'P']:
                if last_result is None:
                    last_result = item
                    streak = 1
                elif item == last_result:
                    streak += 1
                else:
                    break
        
        streak_type = LABEL_MAP.get(last_result, -1)
        prev_result = LABEL_MAP.get(window[-1], -1) if window else -1

        features = [b_ratio, p_ratio, streak, streak_type, prev_result]
        features_list.append(features)
        labels.append(LABEL_MAP[label])

    return np.array(features_list), np.array(labels)

# =============================================================================
# 主要訓練邏輯
# =============================================================================
def train():
    """執行完整的模型訓練流程。"""
    print("="*50)
    print("開始重新訓練 AI 模型...")
    print("="*50)

    # 1. 建立模型儲存目錄
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"✅ 已建立目錄: {MODEL_DIR}")

    # 2. 生成平衡的模擬數據以避免偏差
    print(f"🔄 正在生成 {SYNTHETIC_DATA_SIZE} 筆高品質模擬數據...")
    synthetic_roadmap = []
    # 根據百家樂真實機率 (排除和局後): 閒家勝率約 49.32%, 莊家勝率約 50.68%
    p_win_prob = 0.4932 
    for _ in range(SYNTHETIC_DATA_SIZE):
        if random.random() < p_win_prob:
            synthetic_roadmap.append('P')
        else:
            synthetic_roadmap.append('B')
    print("✅ 模擬數據生成完畢。")

    # 3. 提取基礎特徵
    print("🔄 正在提取基礎特徵...")
    X_basic, y = extract_features_for_training(synthetic_roadmap)

    if len(X_basic) == 0:
        print("❌ 錯誤：無法從數據中提取任何特徵。訓練中止。")
        return
    print(f"✅ 基礎特徵提取完成，共 {len(y)} 筆訓練樣本。")

    # 4. 在完整的序列上訓練 HMM 模型
    print("🔄 正在訓練 HMM 模型...")
    hmm_roadmap_numeric = np.array([LABEL_MAP[r] for r in synthetic_roadmap if r in LABEL_MAP]).reshape(-1, 1)
    
    # 【關鍵修正】: 對於離散觀測值 (莊/閒)，必須使用 CategoricalHMM
    # n_components 是隱藏狀態的數量，4是一個合理的起始值
    hmm_model = hmm.CategoricalHMM(n_components=4, n_iter=100, random_state=42, tol=0.01)
    hmm_model.fit(hmm_roadmap_numeric)
    joblib.dump(hmm_model, os.path.join(MODEL_DIR, 'hmm_model.pkl'))
    print("✅ HMM 模型 (hmm_model.pkl) 已訓練並儲存。")

    # 5. 使用訓練好的 HMM 模型提取進階特徵
    print("🔄 正在使用 HMM 模型提取進階特徵...")
    hmm_features = []
    for i in range(N_FEATURES_WINDOW, len(synthetic_roadmap)):
        if synthetic_roadmap[i] not in LABEL_MAP:
            continue

        current_roadmap_numeric = np.array([LABEL_MAP[r] for r in synthetic_roadmap[:i] if r in LABEL_MAP]).reshape(-1, 1)
        
        if len(current_roadmap_numeric) < 1:
            hmm_features.append([0.5, 0.5]) # 對於太短的序列使用預設機率
            continue
            
        try:
            # 使用與 app.py 相同的穩健預測邏輯
            hidden_states = hmm_model.predict(current_roadmap_numeric)
            last_state = hidden_states[-1]
            transition_probs = hmm_model.transmat_[last_state, :]
            emission_probs = hmm_model.emissionprob_
            
            prob_b = np.dot(transition_probs, emission_probs[:, LABEL_MAP['B']])
            prob_p = np.dot(transition_probs, emission_probs[:, LABEL_MAP['P']])
            
            total_prob = prob_b + prob_p
            if total_prob > 1e-9:
                hmm_features.append([prob_b/total_prob, prob_p/total_prob])
            else:
                hmm_features.append([0.5, 0.5])
        except Exception:
            hmm_features.append([0.5, 0.5]) # 如果出錯則使用預設值
    
    X_combined = np.concatenate([X_basic, np.array(hmm_features)], axis=1)
    print(f"✅ 進階特徵提取完成，最終特徵維度: {X_combined.shape[1]}")

    # 6. 訓練 Scaler 和 XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🔄 正在標準化特徵 (Scaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("✅ 標準化器 (scaler.pkl) 已儲存。")
    
    print("🔄 正在訓練 XGBoost 最終模型...")
    xgb_model = XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        n_estimators=150, learning_rate=0.05, max_depth=4,
        use_label_encoder=False, random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    accuracy = xgb_model.score(X_test_scaled, y_test)
    print(f"📈 模型在測試集上的準確率: {accuracy:.4f}")
    
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    print("✅ XGBoost 模型 (xgb_model.pkl) 已儲存。")
    
    # 7. 儲存特徵資訊
    feature_info = {'use_hmm_features': True}
    joblib.dump(feature_info, os.path.join(MODEL_DIR, 'feature_info.pkl'))
    print("✅ 特徵資訊 (feature_info.pkl) 已儲存。")

    print("\n🎉 所有模型已成功重新訓練並儲存！")

if __name__ == '__main__':
    train()

