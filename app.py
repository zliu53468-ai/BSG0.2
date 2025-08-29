import os
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from hmmlearn import hmm

# =============================================================================
# Flask 應用程式與日誌設定
# =============================================================================

app = Flask(__name__)

# 設定日誌
if not os.path.exists('logs'):
    os.makedirs('logs')

# 建立日誌處理器
log_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=5)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
log_handler.setFormatter(formatter)

# 清除預設處理器並加入我們自己的
if app.logger.hasHandlers():
    app.logger.handlers.clear()
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

# 設定 CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# =============================================================================
# 全域變數與設定
# =============================================================================

HISTORY_FILE = 'history.json'
MODEL_DIR = 'models'
LABEL_MAP = {'B': 0, 'P': 1}
REVERSE_MAP = {0: '莊', 1: '閒'}

# 初始歷史數據
INITIAL_HISTORY_DATA = [
    "P", "P", "T", "B", "T", "B", "P", "B", "P", "P", "B", "B", "T", "B", "B", "P", "B", "B", "P", "B", "B", "T", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "P", "B", "P", "T", "T", "B", "P", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "B", "B", "B", "P", "P", "B", "P", "B", "B", "P", "P", "P", "P", "P", "P", "B", "B", "T", "B", "T", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "T", "T", "B", "B", "B", "B", "P", "B", "B", "B", "P", "B", "T", "P", "P", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "P", "P", "B", "T", "B", "P", "B", "T", "B", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "B", "T", "T", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "T", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "P", "T", "B", "P", "T", "B", "B", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "B", "B", "P", "B", "P", "B", "T", "B", "B", "P", "P", "B", "B", "P", "T", "P", "P", "B", "P", "B", "B", "B", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "T", "T", "B", "P", "P", "B", "P", "P", "B", "B", "P", "B", "P", "T", "P", "P", "P", "P", "B", "B", "B", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "P", "B", "P", "B", "P", "T", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "B", "T", "B", "T", "B", "P", "B", "P", "T", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "P", "P", "B", "P", "P", "P", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "B", "P", "P", "T", "P", "B", "B", "T", "B", "P", "T", "P", "B", "P", "B", "B", "P", "B", "B", "T", "P", "P", "P", "P", "T", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "T", "P", "P", "T", "P", "P", "B", "P", "P", "B", "P", "P", "B", "P", "P", "T", "B", "P", "B", "P", "P", "B", "B", "B", "B", "T", "T", "T", "B", "B", "B", "B", "B", "B", "P", "P", "P", "T", "P", "T", "B", "P", "P", "T", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "P", "P", "T", "P", "B", "P", "P", "B", "T", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "T", "T", "B", "P", "B", "B", "B", "T", "T", "B", "B", "P", "B", "T", "P", "B", "P", "B", "P", "P", "P", "B", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", "B", "B", "P", "B", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "B", "B", "P", "B", "T", "T", "P", "B", "B", "B", "P", "B", "B", "P", "B", "P", "B", "P", "P", "P", "P", "P", "P", "B", "B", "B", "P", "T", "P", "B", "T", "B", "B", "B", "B", "T", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "B", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "B", "B", "B", "T", "B", "B", "P", "B", "P", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "P", "B", "P", "P", "B", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "B", "B", "B", "B", "B", "B", "P", "T", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "P", "P", "T", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "P", "B", "T", "P", "B", "B", "P", "B", "P", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "T", "P", "P", "B", "T", "B", "P", "P", "P", "B", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "B", "P", "P", "T", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "P", "T", "P", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "T", "T", "P", "T", "B", "T", "P", "T", "P", "T", "P", "P", "B", "B", "P", "P", "P", "P", "P"
]

# =============================================================================
# 數據處理函式
# =============================================================================

def load_data():
    """從檔案載入歷史數據，若檔案不存在則使用初始數據。"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            app.logger.info(f"歷史數據檔案 '{HISTORY_FILE}' 不存在，寫入初始數據。")
            save_data(INITIAL_HISTORY_DATA)
            return INITIAL_HISTORY_DATA
    except Exception as e:
        app.logger.error(f"讀取歷史數據失敗: {e}", exc_info=True)
        return INITIAL_HISTORY_DATA

def save_data(data):
    """將數據儲存到檔案。"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        app.logger.error(f"儲存歷史數據失敗: {e}", exc_info=True)

# =============================================================================
# 特徵工程函式
# =============================================================================

def extract_features(roadmap, hmm_model=None, use_hmm_features=False):
    """從路紙中提取特徵，確保特徵維度一致性。"""
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
    hmm_prediction = "等待"

    if use_hmm_features:
        hmm_banker_prob = 0.5
        hmm_player_prob = 0.5
        
        if hmm_model and len(roadmap) > 1:
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
                    
                    hmm_prediction = "莊" if hmm_banker_prob > hmm_player_prob else "閒"
            except Exception as e:
                app.logger.warning(f"HMM 特徵提取失敗: {e}. 使用預設機率。")
        
        features.extend([hmm_banker_prob, hmm_player_prob])
    
    return np.array(features, dtype=np.float32), hmm_prediction

def prepare_training_data(roadmap, hmm_model=None, use_hmm_features=False):
    """準備用於模型訓練的特徵-標籤對。"""
    filtered = [r for r in roadmap if r in LABEL_MAP]
    X, y = [], []
    for i in range(1, len(filtered)):
        current_features, _ = extract_features(filtered[:i], hmm_model, use_hmm_features)
        X.append(current_features)
        y.append(LABEL_MAP[filtered[i]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# =============================================================================
# 模型訓練函式
# =============================================================================

def train_hmm_model(all_history):
    """訓練並儲存 HMM 模型。"""
    hmm_model_path = os.path.join(MODEL_DIR, 'hmm_model.pkl')
    if os.path.exists(hmm_model_path):
        try:
            app.logger.info("HMM 模型已存在，載入現有模型。")
            return joblib.load(hmm_model_path)
        except Exception as e:
            app.logger.error(f"載入現有 HMM 模型失敗: {e}，將重新訓練。")

    app.logger.info("開始訓練 HMM 模型...")
    hmm_observations = np.array([LABEL_MAP[r] for r in all_history if r in LABEL_MAP]).reshape(-1, 1)

    if len(hmm_observations) < 20 or len(np.unique(hmm_observations)) < 2:
        app.logger.warning("HMM 訓練數據不足或缺乏多樣性，跳過 HMM 訓練。")
        return None

    try:
        hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
        hmm_model.fit(hmm_observations)
        
        if hasattr(hmm_model, 'monitor_') and not hmm_model.monitor_.converged:
            app.logger.warning("HMM 模型訓練完成但可能未收斂。")
        else:
             app.logger.info("HMM 模型訓練成功。")

        joblib.dump(hmm_model, hmm_model_path)
        app.logger.info(f"HMM 模型已儲存至 {hmm_model_path}")
        return hmm_model
    except Exception as e:
        app.logger.error(f"HMM 模型訓練失敗: {e}", exc_info=True)
        return None

def train_models_if_needed():
    """應用程式啟動時，檢查並訓練必要的模型。"""
    app.logger.info("檢查模型是否需要訓練...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    all_history = load_data()
    if not all_history:
        app.logger.error("無法載入歷史數據，無法訓練模型。")
        return

    hmm_model = train_hmm_model(all_history)
    use_hmm_features = hmm_model is not None

    xgb_model_path = os.path.join(MODEL_DIR, 'xgb_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(xgb_model_path) or not os.path.exists(scaler_path):
        app.logger.info("XGBoost 模型或 Scaler 不存在，開始訓練...")
        X_train, y_train = prepare_training_data(all_history, hmm_model, use_hmm_features)

        if X_train.shape[0] < 10:
            app.logger.error("訓練數據不足，無法訓練 XGBoost 模型。")
            return
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        joblib.dump(scaler, scaler_path)

        xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb.fit(X_scaled, y_train)
        joblib.dump(xgb, xgb_model_path)
        app.logger.info(f"XGBoost 模型訓練完成並儲存至 {xgb_model_path}")

        feature_info = {'use_hmm_features': use_hmm_features}
        joblib.dump(feature_info, os.path.join(MODEL_DIR, 'feature_info.pkl'))
    else:
        app.logger.info("所有模型檔案均已存在，無需重新訓練。")

# =============================================================================
# **修正**: 在 Gunicorn 啟動時執行模型訓練
# =============================================================================
with app.app_context():
    train_models_if_needed()

# =============================================================================
# Flask 路由 (API Endpoints)
# =============================================================================

@app.route("/", methods=["GET"])
def home():
    """根路由：提供服務狀態訊息。"""
    return jsonify({
        "status": "online",
        "message": "Baccarat AI Prediction Engine is running."
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查路由"""
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    """預測路由：接收路紙數據，回傳預測機率。"""
    try:
        data = request.get_json()
        if not data or "roadmap" not in data:
            return jsonify({"error": "無效的請求，缺少 'roadmap' 欄位。"}), 400

        received_roadmap = [r for r in data["roadmap"] if r in ["B", "P", "T"]]
        
        current_history = load_data()
        if received_roadmap != current_history:
            save_data(received_roadmap)
            app.logger.info(f"歷史數據已更新，新數據長度: {len(received_roadmap)}")

        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
        feature_info = joblib.load(os.path.join(MODEL_DIR, 'feature_info.pkl'))
        use_hmm_features = feature_info.get('use_hmm_features', False)
        
        hmm_model = None
        if use_hmm_features and os.path.exists(os.path.join(MODEL_DIR, 'hmm_model.pkl')):
            hmm_model = joblib.load(os.path.join(MODEL_DIR, 'hmm_model.pkl'))

        features, hmm_prediction = extract_features(received_roadmap, hmm_model, use_hmm_features)
        
        expected_dim = scaler.n_features_in_
        if features.shape[0] != expected_dim:
            msg = f"特徵維度不符 (預期 {expected_dim}, 實際 {features.shape[0]})"
            app.logger.error(msg)
            return jsonify({"error": msg}), 500

        features_scaled = scaler.transform(features.reshape(1, -1))
        xgb_pred_prob = xgb.predict_proba(features_scaled)[0]

        banker_prob = float(xgb_pred_prob[LABEL_MAP['B']])
        player_prob = float(xgb_pred_prob[LABEL_MAP['P']])
        tie_prob = 0.05

        suggestion = "等待"
        if banker_prob > player_prob and banker_prob > 0.52:
            suggestion = "莊"
        elif player_prob > banker_prob and player_prob > 0.52:
            suggestion = "閒"

        return jsonify({
            "banker": round(banker_prob, 4),
            "player": round(player_prob, 4),
            "tie": round(tie_prob, 4),
            "details": {
                "xgb_suggestion": f"{REVERSE_MAP[np.argmax(xgb_pred_prob)]} ({float(np.max(xgb_pred_prob)):.2f})",
                "hmm_suggestion": hmm_prediction,
                "final_suggestion": suggestion
            }
        })

    except FileNotFoundError as e:
        app.logger.error(f"模型檔案未找到: {e}", exc_info=True)
        return jsonify({"error": "模型檔案不存在，請確認模型已訓練。"}), 500
    except Exception as e:
        app.logger.error(f"預測時發生未預期錯誤: {e}", exc_info=True)
        return jsonify({"error": "內部伺服器錯誤"}), 500

# =============================================================================
# 應用程式啟動 (僅供本機開發使用)
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
