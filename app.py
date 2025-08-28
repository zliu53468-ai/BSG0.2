import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

app = Flask(__name__)
CORS(app)

# 檔案路徑設定
HISTORY_FILE = 'history.json'
MODEL_DIR = 'models'

# 初始歷史數據，會在首次啟動時寫入檔案
INITIAL_HISTORY_DATA = [
    "P", "P", "T", "B", "T", "B", "P", "B", "P", "P", "B", "B", "T", "B", "B", "P", "B", "B", "P", "B", "B", "T", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "P", "B", "P", "T", "T", "B", "P", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "B", "B", "B", "P", "P", "B", "P", "B", "B", "P", "P", "P", "P", "P", "P", "B", "B", "T", "B", "T", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "T", "T", "B", "B", "B", "B", "P", "B", "B", "B", "P", "B", "T", "P", "P", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "P", "P", "B", "T", "B", "P", "B", "T", "B", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "B", "T", "T", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "T", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "P", "T", "B", "P", "T", "B", "B", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "B", "B", "P", "B", "P", "B", "T", "B", "B", "P", "P", "B", "B", "P", "T", "P", "P", "B", "P", "B", "B", "B", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "T", "T", "B", "P", "P", "B", "P", "P", "B", "B", "P", "B", "P", "T", "P", "P", "P", "P", "B", "B", "B", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "P", "B", "P", "B", "P", "T", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "B", "T", "B", "T", "B", "P", "B", "P", "T", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "P", "P", "B", "P", "P", "P", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "B", "P", "P", "T", "P", "B", "B", "T", "B", "P", "T", "P", "B", "P", "B", "B", "P", "B", "B", "T", "P", "P", "P", "P", "T", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "T", "P", "P", "T", "P", "P", "B", "P", "P", "B", "P", "P", "B", "P", "P", "T", "B", "P", "B", "P", "P", "B", "B", "B", "B", "T", "T", "T", "B", "B", "B", "B", "B", "B", "P", "P", "P", "T", "P", "T", "B", "P", "P", "T", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "P", "P", "T", "P", "B", "P", "P", "B", "T", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "T", "T", "B", "P", "B", "B", "B", "T", "T", "B", "B", "P", "B", "T", "P", "B", "P", "B", "P", "P", "P", "B", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", "B", "B", "P", "B", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "B", "B", "P", "B", "T", "T", "P", "B", "B", "B", "P", "B", "B", "P", "B", "P", "B", "P", "P", "P", "P", "P", "P", "B", "B", "B", "P", "T", "P", "B", "T", "B", "B", "B", "B", "T", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "B", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "B", "B", "B", "T", "B", "B", "P", "B", "P", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "P", "B", "P", "P", "B", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "B", "B", "B", "B", "B", "B", "P", "T", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "P", "P", "T", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "P", "B", "T", "P", "B", "B", "P", "B", "P", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "T", "P", "P", "B", "T", "B", "P", "P", "P", "B", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "B", "P", "P", "T", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "P", "T", "P", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "T", "T", "P", "T", "B", "T", "P", "T", "P", "T", "P", "P", "B", "B", "P", "P", "P", "P", "P"
]

def load_data():
    """從檔案載入歷史數據"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_data(data):
    """將數據儲存到檔案"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def extract_features(roadmap):
    """從路紙中提取特徵。"""
    N = 20
    window = roadmap[-N:] if len(roadmap) >= N else roadmap[:]
    
    b_count = window.count('莊')
    p_count = window.count('閒')
    total = b_count + p_count
    
    b_ratio = b_count / total if total > 0 else 0.5
    p_ratio = p_count / total if total > 0 else 0.5
    
    streak = 0
    last = None
    for item in reversed(window):
        if item in ['莊', '閒']:
            if last is None:
                last = item
                streak = 1
            elif item == last:
                streak += 1
            else:
                break
    
    streak_type = 0 if last == '莊' else 1 if last == '閒' else -1
    prev = label_map.get(window[-1], -1) if window else -1

    return np.array([b_ratio, p_ratio, streak, streak_type, prev], dtype=np.float32)

def prepare_training_data(roadmap):
    """準備用於模型訓練的數據。"""
    filtered = [r for r in roadmap if r in ['莊', '閒']]
    X = []
    y = []
    for i in range(1, len(filtered)):
        features = extract_features(filtered[:i])
        X.append(features)
        y.append(label_map[filtered[i]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_models():
    """
    這個函式用於離線訓練，請手動或透過排程執行。
    它會讀取所有歷史數據，重新訓練模型並儲存。
    """
    print("開始訓練模型...")
    all_history = load_data()
    if not all_history:
        print("沒有足夠的歷史數據來訓練模型。")
        return

    # 準備訓練資料
    X_train, y_train = prepare_training_data(all_history)
    
    if len(X_train) < 5 or len(np.unique(y_train)) < 2:
        print("訓練資料不足或不平衡，無法進行訓練。")
        return

    # 特徵縮放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    # 訓練與儲存模型
    sgd = SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-4, 
                        learning_rate='adaptive', eta0=0.01, penalty='l2', random_state=42)
    sgd.fit(X_scaled, y_train)
    joblib.dump(sgd, os.path.join(MODEL_DIR, 'sgd_model.pkl'))

    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss',
                        learning_rate=0.1, max_depth=3, random_state=42)
    xgb.fit(X_scaled, y_train)
    joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgb_model.pkl'))

    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=20,
                          max_depth=5, random_state=42)
    lgbm.fit(X_scaled, y_train)
    joblib.dump(lgbm, os.path.join(MODEL_DIR, 'lgbm_model.pkl'))

    print("模型訓練完成並已儲存。")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Multi-Model AI Engine is running."
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    roadmap = data.get("roadmap", []) or data.get("history", [])
    last_valid = [r for r in roadmap if r in ["莊", "閒"]]

    # 接收新數據並保存，但不進行訓練
    if last_valid:
        current_history = load_data()
        # 避免重複寫入
        if not current_history or current_history[-1] != last_valid[-1]:
            current_history.extend(last_valid)
            save_data(current_history)

    # 檢查是否有已訓練好的模型
    model_files = ['sgd_model.pkl', 'xgb_model.pkl', 'lgbm_model.pkl', 'scaler.pkl']
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files):
        return jsonify({
            "banker": 0.5,
            "player": 0.5,
            "tie": 0.05,
            "details": {
                "suggestion": "請先執行模型訓練任務。"
            }
        })
    
    # 載入已訓練好的模型和 Scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    sgd = joblib.load(os.path.join(MODEL_DIR, 'sgd_model.pkl'))
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    lgbm = joblib.load(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))

    # 特徵轉換與預測
    features = extract_features(roadmap).reshape(1, -1)
    features_scaled = scaler.transform(features)

    sgd_pred_prob = sgd.predict_proba(features_scaled)[0]
    xgb_pred_prob = xgb.predict_proba(features_scaled)[0]
    lgb_pred_prob = lgbm.predict_proba(features_scaled)[0]

    sgd_pred = reverse_map[np.argmax(sgd_pred_prob)]
    xgb_pred = reverse_map[np.argmax(xgb_pred_prob)]
    lgb_pred = reverse_map[np.argmax(lgb_pred_prob)]

    # 加權整合
    weight_sgd = 0.2
    weight_xgb = 0.4
    weight_lgbm = 0.4
    total_weights = weight_sgd + weight_xgb + weight_lgbm

    banker_prob = (sgd_pred_prob[0] * weight_sgd +
                   xgb_pred_prob[0] * weight_xgb +
                   lgb_pred_prob[0] * weight_lgbm) / total_weights
                   
    player_prob = (sgd_pred_prob[1] * weight_sgd +
                   xgb_pred_prob[1] * weight_xgb +
                   lgb_pred_prob[1] * weight_lgbm) / total_weights
    
    tie = 0.05
    
    # 綜合建議
    if banker_prob > player_prob and banker_prob > tie:
        suggestion = "莊"
    elif player_prob > banker_prob and player_prob > tie:
        suggestion = "閒"
    else:
        suggestion = "等待"

    return jsonify({
        "banker": round(banker_prob, 3),
        "player": round(player_prob, 3),
        "tie": round(tie, 3),
        "details": {
            "sgd": f"{sgd_pred} ({sgd_pred_prob[np.argmax(sgd_pred_prob)]:.2f})",
            "xgb": f"{xgb_pred} ({xgb_pred_prob[np.argmax(xgb_pred_prob)]:.2f})",
            "lgb": f"{lgb_pred} ({lgb_pred_prob[np.argmax(lgb_pred_prob)]:.2f})",
            "suggestion": suggestion
        }
    })

if __name__ == "__main__":
    # 在第一次啟動時，自動檢查並初始化數據與訓練
    if not os.path.exists(HISTORY_FILE):
        print("首次啟動：偵測到沒有歷史數據檔案，正在自動建立並寫入初始數據...")
        save_data(INITIAL_HISTORY_DATA)
        print("初始數據寫入完成。")

    # 確保模型資料夾存在
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 只有在模型檔案不存在時才進行訓練
    model_files = ['sgd_model.pkl', 'xgb_model.pkl', 'lgbm_model.pkl', 'scaler.pkl']
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files):
        print("偵測到模型檔案不存在，正在自動執行首次訓練...")
        train_models()
        print("首次訓練完成。")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
