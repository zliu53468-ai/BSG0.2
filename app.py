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
# from hmmlearn import hmm # 如果要使用 HMM，請取消註解此行並重新實作 HMM 邏輯

app = Flask(__name__)
CORS(app)

# 檔案路徑設定
HISTORY_FILE = 'history.json'
MODEL_DIR = 'models'

# 標籤映射
label_map = {'莊': 0, '閒': 1}
reverse_map = {0: '莊', 1: '閒'}

# 初始歷史數據，會在首次啟動時寫入檔案
# 'P' 代表閒 (Player), 'B' 代表莊 (Banker), 'T' 代表和 (Tie)
INITIAL_HISTORY_DATA = [
    "P", "P", "T", "B", "T", "B", "P", "B", "P", "P", "B", "B", "T", "B", "B", "P", "B", "B", "P", "B", "B", "T", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "P", "B", "P", "T", "T", "B", "P", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "B", "B", "B", "P", "P", "B", "P", "B", "B", "P", "P", "P", "P", "P", "P", "B", "B", "T", "B", "T", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "T", "T", "B", "B", "B", "B", "P", "B", "B", "B", "P", "B", "T", "P", "P", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "P", "P", "B", "T", "B", "P", "B", "T", "B", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "B", "T", "T", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "T", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "P", "T", "B", "P", "T", "B", "B", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "B", "B", "P", "B", "P", "B", "T", "B", "B", "P", "P", "B", "B", "P", "T", "P", "P", "B", "P", "B", "B", "B", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "T", "T", "B", "P", "P", "B", "P", "P", "B", "B", "P", "B", "P", "T", "P", "P", "P", "P", "B", "B", "B", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "P", "B", "P", "B", "P", "T", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "B", "T", "B", "T", "B", "P", "B", "P", "T", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "P", "P", "B", "P", "P", "P", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "B", "P", "P", "T", "P", "B", "B", "T", "B", "P", "T", "P", "B", "P", "B", "B", "P", "B", "B", "T", "P", "P", "P", "P", "T", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "T", "P", "P", "T", "P", "P", "B", "P", "P", "B", "P", "P", "B", "P", "P", "T", "B", "P", "B", "P", "P", "B", "B", "B", "B", "T", "T", "T", "B", "B", "B", "B", "B", "B", "P", "P", "P", "T", "P", "T", "B", "P", "P", "T", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "P", "P", "T", "P", "B", "P", "P", "B", "T", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "T", "T", "B", "P", "B", "B", "B", "T", "T", "B", "B", "P", "B", "T", "P", "B", "P", "B", "P", "P", "P", "B", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", "B", "B", "P", "B", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "B", "B", "P", "B", "T", "T", "P", "B", "B", "B", "P", "B", "B", "P", "B", "P", "B", "P", "P", "P", "P", "P", "P", "B", "B", "B", "P", "T", "P", "B", "T", "B", "B", "B", "B", "T", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "B", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "B", "B", "B", "T", "B", "B", "P", "B", "P", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "P", "B", "P", "P", "B", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "B", "B", "B", "B", "B", "B", "P", "T", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "P", "P", "T", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "P", "B", "T", "P", "B", "B", "P", "B", "P", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "T", "P", "P", "B", "T", "B", "P", "P", "P", "B", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "B", "P", "P", "T", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "P", "T", "P", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "T", "T", "P", "T", "B", "T", "P", "T", "P", "T", "P", "P", "B", "B", "P", "P", "P", "P", "P"
]

def load_data():
    """從檔案載入歷史數據"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 如果檔案不存在，則寫入初始數據
    print("歷史數據檔案不存在，寫入初始數據。")
    save_data(INITIAL_HISTORY_DATA)
    return INITIAL_HISTORY_DATA

def save_data(data):
    """將數據儲存到檔案"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def extract_features(roadmap):
    """
    從路紙中提取特徵。
    特徵包括：莊/閒比例、連勝次數、連勝類型、前一局結果。
    """
    N = 20 # 考慮最近20局
    window = roadmap[-N:] if len(roadmap) >= N else roadmap[:]
    
    b_count = window.count('B') # 'B' 代表莊
    p_count = window.count('P') # 'P' 代表閒
    total = b_count + p_count
    
    # 避免除以零，若無有效局數則給予預設值
    b_ratio = b_count / total if total > 0 else 0.5
    p_ratio = p_count / total if total > 0 else 0.5
    
    streak = 0
    last = None
    # 計算連勝
    for item in reversed(window):
        if item in ['B', 'P']: # 只考慮莊閒結果
            if last is None:
                last = item
                streak = 1
            elif item == last:
                streak += 1
            else:
                break # 連勝中斷
    
    streak_type = 0 if last == 'B' else 1 if last == 'P' else -1 # 莊為0, 閒為1, 無連勝為-1
    prev = label_map.get(window[-1], -1) if window else -1 # 前一局結果

    return np.array([b_ratio, p_ratio, streak, streak_type, prev], dtype=np.float32)

def prepare_training_data(roadmap):
    """
    準備用於模型訓練的數據。
    將路紙轉換為特徵-標籤對。
    """
    filtered = [r for r in roadmap if r in ['B', 'P']]
    X = []
    y = []
    # 從第二局開始，用前 i 局的數據作為特徵，預測第 i 局的結果
    for i in range(1, len(filtered)):
        features = extract_features(filtered[:i])
        X.append(features)
        y.append(label_map[filtered[i]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_models_if_needed():
    """
    檢查模型檔案是否存在，若不存在則訓練並儲存模型。
    """
    print("開始檢查模型是否需要訓練...")
    model_files = ['sgd_model.pkl', 'xgb_model.pkl', 'lgbm_model.pkl', 'scaler.pkl']
    all_models_exist = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files)

    if not all_models_exist:
        print("偵測到模型檔案不存在，正在自動執行首次訓練...")
        # 確保模型資料夾存在
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        all_history = load_data() # 確保在訓練前載入所有歷史數據
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
    else:
        print("模型檔案已存在，無需重新訓練。")


# 在應用程式啟動時立即執行模型檢查和訓練
train_models_if_needed()


@app.route("/", methods=["GET"])
def home():
    """根路由：提供服務狀態訊息。"""
    return jsonify({
        "status": "success",
        "message": "Multi-Model AI Engine is running."
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    預測路由：接收路紙數據，保存新數據，並使用預訓練模型進行預測。
    """
    data = request.get_json()
    # 接收從前端傳來的完整路紙序列
    received_roadmap = data.get("roadmap", []) or data.get("history", [])
    
    # 過濾出有效的莊閒和結果，只保留 'B', 'P', 'T'
    filtered_received_roadmap = [r for r in received_roadmap if r in ["B", "P", "T"]]

    # --- 數據保存邏輯 ---
    # 假設 filtered_received_roadmap 是前端傳來「最新且完整」的遊戲歷史。
    # 後端將直接使用此數據作為最新的歷史記錄。
    current_history_from_file = load_data()
    
    # 如果接收到的路紙與現有歷史數據不同，則更新歷史檔案
    if filtered_received_roadmap != current_history_from_file:
        save_data(filtered_received_roadmap) # 直接用前端提供的完整路紙覆蓋
        print(f"歷史數據已更新為：{filtered_received_roadmap}")
    else:
        print("收到的路紙與現有歷史數據一致，無需更新。")
    # --- 數據保存邏輯結束 ---
    
    # 實際用於特徵提取和預測的數據 (只考慮 'B'/'P'，因為模型只預測莊閒)
    current_game_sequence_for_prediction = [r for r in filtered_received_roadmap if r in ["B", "P"]]


    # 檢查是否有已訓練好的模型 (此處為運行時檢查，主要用於防止訓練失敗導致的錯誤)
    model_files = ['sgd_model.pkl', 'xgb_model.pkl', 'lgbm_model.pkl', 'scaler.pkl']
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files):
        print("警告: 預測時發現模型檔案缺失，回傳預設機率。請檢查服務啟動日誌中的訓練過程。")
        return jsonify({
            "banker": 0.5,
            "player": 0.5,
            "tie": 0.05,
            "details": {
                "suggestion": "請先執行模型訓練任務。" # 此處提示前端，但實際已嘗試在啟動時訓練
            }
        })
    
    # 載入已訓練好的模型和 Scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    sgd = joblib.load(os.path.join(MODEL_DIR, 'sgd_model.pkl'))
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    lgbm = joblib.load(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))

    # 特徵轉換與預測
    # 這裡使用 `current_game_sequence_for_prediction` 來提取特徵
    features = extract_features(current_game_sequence_for_prediction).reshape(1, -1)
    # 如果沒有足夠的有效歷史數據來提取特徵，可能導致錯誤
    if features.size == 0:
        print("警告: 沒有足夠的莊閒結果來提取特徵，回傳預設機率。")
        return jsonify({
            "banker": 0.5,
            "player": 0.5,
            "tie": 0.05,
            "details": {
                "suggestion": "請輸入更多莊閒結果以進行預測。"
            }
        })

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
    
    # 假定和局機率，可以根據歷史數據調整
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

# 原有的 `if __name__ == "__main__":` 區塊已簡化，僅用於本地開發測試
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
