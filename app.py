import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from hmmlearn import hmm

app = Flask(__name__)
CORS(app)

# 檔案路徑設定
HISTORY_FILE = 'history.json'
MODEL_DIR = 'models'

# 標籤映射
label_map = {'B': 0, 'P': 1}
reverse_map = {0: '莊', 1: '閒'}

# 初始歷史數據
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

def extract_features(roadmap, hmm_model=None, use_hmm_features=False):
    """
    從路紙中提取特徵。
    特徵包括：莊/閒比例、連勝次數、連勝類型、前一局結果。
    如果 use_hmm_features 為 True 且提供了 HMM 模型，則會額外提取 HMM 預測的下一局莊閒機率作為特徵。
    """
    N = 20  # 考慮最近20局
    window = roadmap[-N:] if len(roadmap) >= N else roadmap[:]
    
    b_count = window.count('B')  # 'B' 代表莊
    p_count = window.count('P')  # 'P' 代表閒
    total = b_count + p_count
    
    # 避免除以零，若無有效局數則給予預設值
    b_ratio = b_count / total if total > 0 else 0.5
    p_ratio = p_count / total if total > 0 else 0.5
    
    streak = 0
    last = None
    # 計算連勝
    for item in reversed(window):
        if item in ['B', 'P']:  # 只考慮莊閒結果
            if last is None:
                last = item
                streak = 1
            elif item == last:
                streak += 1
            else:
                break  # 連勝中斷
    
    streak_type = 0 if last == 'B' else 1 if last == 'P' else -1  # 莊為0, 閒為1, 無連勝為-1
    prev = label_map.get(window[-1], -1) if window else -1  # 前一局結果

    features = [b_ratio, p_ratio, streak, streak_type, prev]

    # HMM 特徵提取
    hmm_banker_prob = 0.5
    hmm_player_prob = 0.5
    hmm_prediction = "等待"  # HMM 的獨立預測結果

    # 只有當 use_hmm_features 為 True 且 HMM 模型存在時才提取 HMM 特徵
    if use_hmm_features and hmm_model and hasattr(hmm_model, 'emissionprob_') and len(roadmap) > 1:
        try:
            # 準備 HMM 觀察序列 (只考慮 'B'/'P' 並轉換為數字)
            hmm_observations = np.array([label_map[r] for r in roadmap if r in ['B', 'P']]).reshape(-1, 1)
            
            if len(hmm_observations) > 1:
                # 獲取當前序列最可能的隱藏狀態
                hidden_states = hmm_model.predict(hmm_observations)
                last_hidden_state = hidden_states[-1]
                
                # 獲取發射概率
                emission_probs = hmm_model.emissionprob_[last_hidden_state]
                
                # 由於我們只有兩個觀察值 (0=莊, 1=閒)
                hmm_banker_prob = emission_probs[0]
                hmm_player_prob = emission_probs[1]
                
                # 確保概率和為 1
                total_prob = hmm_banker_prob + hmm_player_prob
                if total_prob > 0:
                    hmm_banker_prob /= total_prob
                    hmm_player_prob /= total_prob
                
                # 根據概率決定預測
                if hmm_banker_prob > hmm_player_prob:
                    hmm_prediction = "莊"
                elif hmm_player_prob > hmm_banker_prob:
                    hmm_prediction = "閒"
                else:
                    hmm_prediction = "等待"
                    
        except Exception as e:
            print(f"HMM 預測失敗: {e}")
            hmm_banker_prob = 0.5
            hmm_player_prob = 0.5
            hmm_prediction = "等待"
    
    # 只有在使用 HMM 特徵時才添加它們
    if use_hmm_features:
        features.extend([hmm_banker_prob, hmm_player_prob])
    
    return np.array(features, dtype=np.float32), hmm_prediction

def prepare_training_data(roadmap, hmm_model=None, use_hmm_features=False):
    """
    準備用於模型訓練的數據。
    將路紙轉換為特徵-標籤對。
    如果 use_hmm_features 為 True 且提供了 HMM 模型，則會將 HMM 預測機率作為額外特徵。
    """
    filtered = [r for r in roadmap if r in ['B', 'P']]
    X = []
    y = []
    # 從第二局開始，用前 i 局的數據作為特徵，預測第 i 局的結果
    for i in range(1, len(filtered)):
        current_features, _ = extract_features(filtered[:i], hmm_model, use_hmm_features)
        X.append(current_features)
        y.append(label_map[filtered[i]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_hmm_model(all_history):
    """
    訓練並儲存 HMM 模型。
    使用 GaussianHMM，因為我們的觀察值是連續的數值。
    """
    hmm_model_path = os.path.join(MODEL_DIR, 'hmm_model.pkl')
    if os.path.exists(hmm_model_path):
        print("HMM 模型檔案已存在，載入現有模型。")
        try:
            return joblib.load(hmm_model_path)
        except:
            print("載入現有 HMM 模型失敗，將重新訓練。")
    
    print("開始訓練 HMM 模型...")
    
    # 準備觀察序列
    hmm_observations = np.array([label_map[r] for r in all_history if r in ['B', 'P']]).reshape(-1, 1)
    
    if len(hmm_observations) < 10:
        print("HMM 訓練數據不足 (至少需要10個莊閒結果)，跳過 HMM 訓練。")
        return None
    
    # 檢查數據多樣性
    if len(np.unique(hmm_observations)) < 2:
        print("HMM 訓練數據缺乏多樣性 (沒有足夠的莊閒變化)，跳過 HMM 訓練。")
        return None
    
    try:
        # 使用 GaussianHMM
        # 設置 2 個隱藏狀態 (莊趨勢, 閒趨勢)
        hmm_model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        
        # 訓練模型
        hmm_model.fit(hmm_observations)
        
        # 檢查模型是否收斂
        if hasattr(hmm_model, 'monitor_') and hmm_model.monitor_.converged:
            print("HMM 模型訓練成功並已收斂。")
        else:
            print("HMM 模型訓練完成但可能未收斂。")
        
        # 保存模型
        joblib.dump(hmm_model, hmm_model_path)
        print("HMM 模型訓練完成並已儲存。")
        return hmm_model
        
    except Exception as e:
        print(f"HMM 模型訓練失敗: {e}")
        return None

def train_models_if_needed():
    """
    檢查模型檔案是否存在，若不存在則訓練並儲存模型。
    """
    print("開始檢查模型是否需要訓練...")
    
    # 確保模型資料夾存在
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 載入所有歷史數據
    all_history = load_data()
    if not all_history:
        print("沒有足夠的歷史數據來訓練模型。")
        return

    # 1. 訓練或載入 HMM 模型
    hmm_model = train_hmm_model(all_history)
    use_hmm_features = hmm_model is not None and hasattr(hmm_model, 'emissionprob_')
    
    # 2. 檢查 XGBoost 模型是否需要訓練
    model_files = ['xgb_model.pkl', 'scaler.pkl']
    all_models_exist = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files)

    if not all_models_exist:
        print("偵測到模型檔案不存在，正在自動執行訓練...")
        
        # 準備訓練資料
        X_train, y_train = prepare_training_data(all_history, hmm_model, use_hmm_features)
        
        # 檢查訓練數據
        if len(X_train) == 0 or len(y_train) == 0:
            print("訓練數據不足，無法進行訓練。")
            return
        
        # 檢查特徵維度是否一致
        expected_feature_dim = 7 if use_hmm_features else 5
        if X_train.shape[1] != expected_feature_dim:
            print(f"錯誤: 訓練數據特徵維度不符 (預期 {expected_feature_dim}, 實際 {X_train.shape[1]})。無法訓練 XGBoost。")
            return
        
        # 特徵縮放
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

        # 訓練與儲存 XGBoost 模型
        xgb = XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        xgb.fit(X_scaled, y_train)
        joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgb_model.pkl'))

        print("XGBoost 模型訓練完成並已儲存。")
        
        # 儲存特徵使用情況
        feature_info = {
            'use_hmm_features': use_hmm_features,
            'feature_dim': expected_feature_dim
        }
        joblib.dump(feature_info, os.path.join(MODEL_DIR, 'feature_info.pkl'))
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
    received_roadmap = data.get("roadmap", []) or data.get("history", [])
    
    # 過濾出有效的莊閒和結果
    filtered_received_roadmap = [r for r in received_roadmap if r in ["B", "P", "T"]]

    # 數據保存邏輯
    current_history_from_file = load_data()
    
    if filtered_received_roadmap != current_history_from_file:
        save_data(filtered_received_roadmap)
        print(f"歷史數據已更新為：{filtered_received_roadmap}")
    else:
        print("收到的路紙與現有歷史數據一致，無需更新。")
    
    # 實際用於特徵提取和預測的數據
    current_game_sequence = [r for r in filtered_received_roadmap if r in ["B", "P"]]

    # 檢查模型檔案是否存在
    model_files = ['xgb_model.pkl', 'scaler.pkl']
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files):
        print("警告: 模型檔案缺失，回傳預設機率。")
        return jsonify({
            "banker": 0.5,
            "player": 0.5,
            "tie": 0.05,
            "details": {
                "xgb": "N/A", "hmm": "N/A",
                "suggestion": "請先執行模型訓練任務。"
            }
        })
    
    # 載入模型和 Scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    
    # 嘗試載入 HMM 模型和特徵信息
    hmm_model = None
    use_hmm_features = False
    hmm_model_path = os.path.join(MODEL_DIR, 'hmm_model.pkl')
    feature_info_path = os.path.join(MODEL_DIR, 'feature_info.pkl')
    
    if os.path.exists(hmm_model_path):
        try:
            hmm_model = joblib.load(hmm_model_path)
        except:
            print("載入 HMM 模型失敗。")
    
    if os.path.exists(feature_info_path):
        try:
            feature_info = joblib.load(feature_info_path)
            use_hmm_features = feature_info.get('use_hmm_features', False)
        except:
            print("載入特徵信息失敗。")

    # 特徵轉換與預測
    features, hmm_prediction = extract_features(current_game_sequence, hmm_model, use_hmm_features)
    
    # 檢查特徵維度
    expected_feature_size = 7 if use_hmm_features else 5
    if len(features) != expected_feature_size:
        print(f"警告: 特徵數量不符 (預期 {expected_feature_size}, 實際 {len(features)})，回傳預設機率。")
        return jsonify({
            "banker": 0.5,
            "player": 0.5,
            "tie": 0.05,
            "details": {
                "xgb": "N/A", "hmm": "N/A",
                "suggestion": "請輸入更多莊閒結果以進行預測。"
            }
        })

    features_scaled = scaler.transform(features.reshape(1, -1))
    xgb_pred_prob = xgb.predict_proba(features_scaled)[0]

    # 計算概率
    banker_prob = xgb_pred_prob[0]
    player_prob = xgb_pred_prob[1]
    tie = 0.05  # 假定和局機率
    
    # 綜合建議
    if banker_prob > player_prob and banker_prob > tie:
        suggestion = "莊"
    elif player_prob > banker_prob and player_prob > tie:
        suggestion = "閒"
    else:
        suggestion = "等待"

    return jsonify({
        "banker": float(round(banker_prob, 3)),
        "player": float(round(player_prob, 3)),
        "tie": float(round(tie, 3)),
        "details": {
            "xgb": f"{reverse_map[np.argmax(xgb_pred_prob)]} ({float(np.max(xgb_pred_prob)):.2f})",
            "hmm": hmm_prediction,
            "suggestion": suggestion
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
