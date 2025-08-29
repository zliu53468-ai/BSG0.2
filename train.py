import os
import json
import logging
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from hmmlearn import hmm

# =============================================================================
# 訓練腳本設定
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HISTORY_FILE = 'history.json'
MODEL_DIR = 'models'
LABEL_MAP = {'B': 0, 'P': 1}

INITIAL_HISTORY_DATA = [
    "P", "P", "T", "B", "T", "B", "P", "B", "P", "P", "B", "B", "T", "B", "B", "P", "B", "B", "P", "B", "B", "T", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "P", "B", "P", "T", "T", "B", "P", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "B", "B", "B", "P", "P", "B", "P", "B", "B", "P", "P", "P", "P", "P", "P", "B", "B", "T", "B", "T", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "T", "T", "B", "B", "B", "B", "P", "B", "B", "B", "P", "B", "T", "P", "P", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "P", "P", "B", "T", "B", "P", "B", "T", "B", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "B", "T", "T", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "T", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "P", "T", "B", "P", "T", "B", "B", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "B", "B", "P", "B", "P", "B", "T", "B", "B", "P", "P", "B", "B", "P", "T", "P", "P", "B", "P", "B", "B", "B", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "T", "T", "B", "P", "P", "B", "P", "P", "B", "B", "P", "B", "P", "T", "P", "P", "P", "P", "B", "B", "B", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "P", "B", "P", "B", "P", "T", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "B", "T", "B", "T", "B", "P", "B", "P", "T", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "P", "P", "B", "P", "P", "P", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "B", "P", "P", "T", "P", "B", "B", "T", "B", "P", "T", "P", "B", "P", "B", "B", "P", "B", "B", "T", "P", "P", "P", "P", "T", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "T", "P", "P", "T", "P", "P", "B", "P", "P", "B", "P", "P", "B", "P", "P", "T", "B", "P", "B", "P", "P", "B", "B", "B", "B", "T", "T", "T", "B", "B", "B", "B", "B", "B", "P", "P", "P", "T", "P", "T", "B", "P", "P", "T", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "P", "P", "T", "P", "B", "P", "P", "B", "T", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "T", "T", "B", "P", "B", "B", "B", "T", "T", "B", "B", "P", "B", "T", "P", "B", "P", "B", "P", "P", "P", "B", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", "B", "B", "P", "B", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "B", "B", "P", "B", "T", "T", "P", "B", "B", "B", "P", "B", "B", "P", "B", "P", "B", "P", "P", "P", "P", "P", "P", "B", "B", "B", "P", "T", "P", "B", "T", "B", "B", "B", "B", "T", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "B", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "B", "B", "B", "T", "B", "B", "P", "B", "P", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "P", "B", "P", "P", "B", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "B", "B", "B", "B", "B", "B", "P", "T", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "P", "P", "T", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "P", "B", "T", "P", "B", "B", "P", "B", "P", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "T", "P", "P", "B", "T", "B", "P", "P", "P", "B", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "B", "P", "P", "T", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "P", "T", "P", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "T", "T", "P", "T", "B", "T", "P", "T", "P", "T", "P", "P", "B", "B", "P", "P", "P", "P", "P"
]

# =============================================================================
# 數據處理函式
# =============================================================================

def load_data():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    logging.info(f"歷史檔案 '{HISTORY_FILE}' 不存在，將使用初始數據。")
    return INITIAL_HISTORY_DATA

def save_data(data):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# =============================================================================
# 特徵工程函式
# =============================================================================

def extract_features(roadmap, hmm_model=None, use_hmm_features=False):
    N = 20
    window = roadmap[-N:]
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
    if use_hmm_features:
        hmm_banker_prob = 0.5
        hmm_player_prob = 0.5
        # **修正**: 增加 hasattr 檢查，確保模型是成功訓練的
        if hmm_model and hasattr(hmm_model, 'emissionprob_') and len(roadmap) > 1:
            try:
                hmm_observations = np.array([LABEL_MAP[r] for r in roadmap if r in LABEL_MAP]).reshape(-1, 1)
                if len(hmm_observations) > 1:
                    hidden_states = hmm_model.predict(hmm_observations)
                    last_hidden_state = hidden_states[-1]
                    emission_probs = hmm_model.emissionprob_[last_hidden_state]
                    hmm_banker_prob = emission_probs[LABEL_MAP['B']]
                    hmm_player_prob = emission_probs[LABEL_MAP['P']]
                    total_prob = hmm_banker_prob + hmm_player_prob
                    if total_prob > 0:
                        hmm_banker_prob /= total_prob
                        hmm_player_prob /= total_prob
            except Exception as e:
                logging.warning(f"HMM 特徵提取失敗: {e}.")
        features.extend([hmm_banker_prob, hmm_player_prob])
    return np.array(features, dtype=np.float32)

def prepare_training_data(roadmap, hmm_model=None, use_hmm_features=False):
    filtered = [r for r in roadmap if r in LABEL_MAP]
    X, y = [], []
    for i in range(1, len(filtered)):
        current_features = extract_features(filtered[:i], hmm_model, use_hmm_features)
        X.append(current_features)
        y.append(LABEL_MAP[filtered[i]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# =============================================================================
# 模型訓練函式
# =============================================================================

def train_hmm_model(all_history):
    hmm_model_path = os.path.join(MODEL_DIR, 'hmm_model.pkl')
    logging.info("開始訓練 HMM 模型...")
    hmm_observations = np.array([LABEL_MAP[r] for r in all_history if r in LABEL_MAP]).reshape(-1, 1)
    if len(hmm_observations) < 20 or len(np.unique(hmm_observations)) < 2:
        logging.warning("HMM 訓練數據不足，跳過訓練。")
        return None
    try:
        hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
        hmm_model.fit(hmm_observations)
        # **修正**: 檢查模型是否真的收斂並產生了必要的屬性
        if not hasattr(hmm_model, 'emissionprob_'):
            logging.error("HMM 模型訓練後未產生 'emissionprob_' 屬性，訓練可能失敗。")
            return None
        joblib.dump(hmm_model, hmm_model_path)
        logging.info(f"HMM 模型已儲存至 {hmm_model_path}")
        return hmm_model
    except Exception as e:
        logging.error(f"HMM 模型訓練失敗: {e}", exc_info=True)
        return None

def train_models():
    """執行所有模型的訓練流程。"""
    logging.info("檢查並開始模型訓練...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(HISTORY_FILE):
        save_data(INITIAL_HISTORY_DATA)

    all_history = load_data()
    if not all_history:
        logging.error("歷史數據為空，無法訓練模型。")
        return

    hmm_model = train_hmm_model(all_history)
    use_hmm_features = hmm_model is not None

    logging.info("準備訓練 XGBoost 模型...")
    X_train, y_train = prepare_training_data(all_history, hmm_model, use_hmm_features)

    if X_train.shape[0] < 10:
        logging.error("XGBoost 訓練數據不足，無法訓練。")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_scaled, y_train)
    joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    logging.info("XGBoost 模型訓練完成並儲存。")

    feature_info = {'use_hmm_features': use_hmm_features}
    joblib.dump(feature_info, os.path.join(MODEL_DIR, 'feature_info.pkl'))
    logging.info("特徵資訊已儲存。")

# =============================================================================
# 主執行區塊
# =============================================================================
if __name__ == "__main__":
    train_models()
