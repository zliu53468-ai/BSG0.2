# -*- coding: utf-8 -*-
import os
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# =============================================================================
# Flask 應用程式與日誌設定
# =============================================================================
app = Flask(__name__)
if not os.path.exists('logs'):
    os.makedirs('logs')
log_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=5)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
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
N_FEATURES_WINDOW = 20

# =============================================================================
# 數據處理與特徵函式
# =============================================================================
def load_data():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        app.logger.error(f"讀取歷史數據失敗: {e}", exc_info=True)
    return []

def save_data(data):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        app.logger.error(f"儲存歷史數據失敗: {e}", exc_info=True)

def predict_hmm_next_step(hmm_model, roadmap_numeric):
    try:
        hidden_states = hmm_model.predict(roadmap_numeric)
        last_state = hidden_states[-1]
        transition_probs = hmm_model.transmat_[last_state, :]
        emission_probs = hmm_model.emissionprob_
        prob_b = np.dot(transition_probs, emission_probs[:, LABEL_MAP['B']])
        prob_p = np.dot(transition_probs, emission_probs[:, LABEL_MAP['P']])
        
        total_prob = prob_b + prob_p
        if total_prob > 1e-9:
            prob_b /= total_prob
            prob_p /= total_prob
        else:
            return 0.5, 0.5, "觀望"

        # 【安全機制】如果機率太接近，視為趨勢不明，返回"觀望"
        if abs(prob_b - prob_p) < 0.02:
            prediction = "觀望"
        else:
            prediction = "莊" if prob_b > prob_p else "閒"
        return prob_b, prob_p, prediction
    except Exception as e:
        app.logger.warning(f"HMM 預測時發生錯誤: {e}. 返回預設值。")
        return 0.5, 0.5, "觀望"

def extract_features_for_prediction(roadmap, hmm_model):
    window = roadmap[-N_FEATURES_WINDOW:]
    b_count = window.count('B')
    p_count = window.count('P')
    total = b_count + p_count
    b_ratio = b_count / total if total > 0 else 0.5
    p_ratio = p_count / total if total > 0 else 0.5
    streak = 0
    last_result = None
    for item in reversed(window):
        if item in ['B', 'P']:
            if last_result is None: last_result = item; streak = 1
            elif item == last_result: streak += 1
            else: break
    streak_type = LABEL_MAP.get(last_result, -1)
    prev_result = LABEL_MAP.get(window[-1], -1) if window else -1
    features_basic = [b_ratio, p_ratio, streak, streak_type, prev_result]
    
    hmm_prediction = "觀望"
    hmm_banker_prob, hmm_player_prob = 0.5, 0.5
    if hmm_model:
        roadmap_numeric = np.array([LABEL_MAP[r] for r in roadmap if r in LABEL_MAP]).reshape(-1, 1)
        if len(roadmap_numeric) > 1:
            hmm_banker_prob, hmm_player_prob, hmm_prediction = predict_hmm_next_step(hmm_model, roadmap_numeric)

    features_combined = features_basic + [hmm_banker_prob, hmm_player_prob]
    return np.array(features_combined, dtype=np.float32), hmm_prediction

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

        if len(received_roadmap) < N_FEATURES_WINDOW:
             return jsonify({
                "banker": 0.5, "player": 0.5, "tie": 0.05,
                "details": {"xgb_suggestion": "觀望", "hmm_suggestion": "觀望", "final_suggestion": "數據不足"}
            })

        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
        hmm_model = joblib.load(os.path.join(MODEL_DIR, 'hmm_model.pkl'))

        features, hmm_prediction = extract_features_for_prediction(received_roadmap, hmm_model)
        
        features_scaled = scaler.transform(features.reshape(1, -1))
        xgb_pred_prob = xgb.predict_proba(features_scaled)[0]
        xgb_suggestion = REVERSE_MAP[np.argmax(xgb_pred_prob)]

        banker_prob = float(xgb_pred_prob[LABEL_MAP['B']])
        player_prob = float(xgb_pred_prob[LABEL_MAP['P']])
        tie_prob = 1.0 - (banker_prob + player_prob)

        return jsonify({
            "banker": round(banker_prob, 4),
            "player": round(player_prob, 4),
            "tie": round(tie_prob, 4),
            "details": {
                "xgb_suggestion": xgb_suggestion,
                "hmm_suggestion": hmm_prediction,
            }
        })
    except FileNotFoundError as e:
        app.logger.error(f"模型檔案未找到: {e}", exc_info=True)
        return jsonify({"error": "模型檔案不存在，請確認模型已訓練。"}), 500
    except Exception as e:
        app.logger.error(f"預測時發生未預期錯誤: {e}", exc_info=True)
        return jsonify({"error": "內部伺服器錯誤"}), 500

# =============================================================================
# 應用程式啟動
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)


