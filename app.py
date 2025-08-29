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

log_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=5)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
log_handler.setFormatter(formatter)

if app.logger.hasHandlers():
    app.logger.handlers.clear()
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

CORS(app, resources={r"/*": {"origins": "*"}})

# =============================================================================
# 全域變數與設定
# =============================================================================

HISTORY_FILE = 'history.json'
MODEL_DIR = 'models'
LABEL_MAP = {'B': 0, 'P': 1}
REVERSE_MAP = {0: '莊', 1: '閒'}

# =============================================================================
# 數據與特徵處理函式 (僅供預測使用)
# =============================================================================

def load_data():
    """從檔案載入歷史數據。"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        app.logger.error(f"讀取歷史數據失敗: {e}", exc_info=True)
    return []

def save_data(data):
    """將數據儲存到檔案。"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        app.logger.error(f"儲存歷史數據失敗: {e}", exc_info=True)

def extract_features(roadmap, hmm_model=None, use_hmm_features=False):
    """從路紙中提取特徵。"""
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
                    
                    hmm_prediction = "莊" if hmm_banker_prob > hmm_player_prob else "閒"
            except Exception as e:
                app.logger.warning(f"HMM 特徵提取失敗: {e}. 使用預設機率。")
        
        features.extend([hmm_banker_prob, hmm_player_prob])
    
    return np.array(features, dtype=np.float32), hmm_prediction

# =============================================================================
# Flask 路由 (API Endpoints)
# =============================================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "message": "Baccarat AI Prediction Engine is running."})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
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
