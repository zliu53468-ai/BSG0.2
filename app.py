import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from hmmlearn import hmm # 啟用 HMM

app = Flask(__name__)
CORS(app)

# 檔案路徑設定
HISTORY_FILE = 'history.json'
MODEL_DIR = 'models'

# 標籤映射
label_map = {'B': 0, 'P': 1}
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

def extract_features(roadmap, hmm_model=None):
    """
    從路紙中提取特徵。
    特徵包括：莊/閒比例、連勝次數、連勝類型、前一局結果。
    如果提供了 HMM 模型，則會額外提取 HMM 預測的下一局莊閒機率作為特徵。
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

    features = [b_ratio, p_ratio, streak, streak_type, prev]

    # --- HMM 特徵提取 ---
    hmm_banker_prob = 0.5
    hmm_player_prob = 0.5
    hmm_prediction = "等待" # HMM 的獨立預測結果

    # 只有當 HMM 模型存在且已成功訓練時才嘗試獲取 emissionprob_
    if hmm_model and hasattr(hmm_model, 'emissionprob_') and len(roadmap) > 1: # HMM 至少需要兩個觀察值才能預測轉換
        # 準備 HMM 觀察序列 (只考慮 'B'/'P' 並轉換為數字)
        hmm_observations_for_prediction = np.array([label_map[r] for r in roadmap if r in ['B', 'P']]).reshape(-1, 1)
        
        if hmm_observations_for_prediction.size > 0:
            try:
                # 獲取當前序列最可能的隱藏狀態
                # 使用 `predict` 找到最可能的隱藏狀態序列
                hidden_states = hmm_model.predict(hmm_observations_for_prediction)
                last_hidden_state = hidden_states[-1]
                
                # 根據最後一個隱藏狀態的發射機率來預測下一觀察值
                # 注意：GMMHMM 的 emissionprob_ 是一個列表，每個元素是 GMM 的 weights_
                # 我們需要從 GMM 的 means_ 和 covars_ 計算發射機率
                # 這裡假設 GMMHMM 的發射機率是簡單的 GMM 權重
                # 為了簡化，直接使用 GMMHMM 的 emissionprob_ 屬性（如果存在且結構兼容）
                # 由於 GaussianHMM 的 emissionprob_ 是 (n_components, n_observations_types)
                # 而 GMMHMM 的 emissionprob_ 結構更複雜 (n_components, n_mix, n_features)
                # 我們需要調整這裡的訪問方式
                
                # 更穩健的 HMM emissionprob_ 訪問方式 (適用於 GaussianHMM)
                if isinstance(hmm_model, hmm.GaussianHMM):
                    hmm_banker_prob = float(hmm_model.emissionprob_[last_hidden_state, 0]) # 狀態發射 'B' (0) 的機率
                    hmm_player_prob = float(hmm_model.emissionprob_[last_hidden_state, 1]) # 狀態發射 'P' (1) 的機率
                elif isinstance(hmm_model, hmm.GMMHMM):
                    # 對於 GMMHMM，emissionprob_ 是一個 GMM 對象的列表
                    # 我們需要從 GMM 對象中提取每個觀察值的機率
                    # 這裡簡化為使用 GMM 的權重作為機率，這可能不完全準確，但作為特徵是可行的
                    # 或者更複雜地計算 GMM 的 PDF
                    # 為了簡化，我們直接使用 GMMHMM 的 emissionprob_ 屬性 (如果它是一個數組)
                    # 如果 GMMHMM 的 emissionprob_ 是一個 GMM 對象的列表，則需要更複雜的處理
                    # 暫時回退到預設值，並在日誌中提醒
                    print("警告: GMMHMM 的 emissionprob_ 訪問方式需要更複雜的處理，暫時使用預設機率。")
                    hmm_banker_prob = 0.5
                    hmm_player_prob = 0.5
                else:
                    hmm_banker_prob = 0.5
                    hmm_player_prob = 0.5


                # 確保機率和為 1
                total_hmm_prob = hmm_banker_prob + hmm_player_prob
                if total_hmm_prob > 0:
                    hmm_banker_prob /= total_hmm_prob
                    hmm_player_prob /= total_hmm_prob
                else: # 避免除以零
                    hmm_banker_prob = 0.5
                    hmm_player_prob = 0.5

                if hmm_banker_prob > hmm_player_prob:
                    hmm_prediction = "莊"
                elif hmm_player_prob > hmm_banker_prob:
                    hmm_prediction = "閒"
                else:
                    hmm_prediction = "等待"

            except Exception as e:
                print(f"HMM 預測失敗: {e}")
                # HMM 預測失敗時，使用預設值
                hmm_banker_prob = 0.5
                hmm_player_prob = 0.5
                hmm_prediction = "等待"
    
    features.extend([hmm_banker_prob, hmm_player_prob]) # 將 HMM 預測機率作為新特徵
    
    return np.array(features, dtype=np.float32), hmm_prediction # 返回 HMM 的獨立預測結果

def prepare_training_data(roadmap, hmm_model=None):
    """
    準備用於模型訓練的數據。
    將路紙轉換為特徵-標籤對。
    如果提供了 HMM 模型，則會將 HMM 預測機率作為額外特徵。
    """
    filtered = [r for r in roadmap if r in ['B', 'P']]
    X = []
    y = []
    # 從第二局開始，用前 i 局的數據作為特徵，預測第 i 局的結果
    for i in range(1, len(filtered)):
        # 注意: 這裡傳遞的是 `filtered[:i]`，這是一個子序列，HMM 應該基於這個子序列進行預測
        current_features, _ = extract_features(filtered[:i], hmm_model) 
        X.append(current_features)
        y.append(label_map[filtered[i]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_hmm_model(all_history):
    """
    訓練並儲存 HMM 模型。
    """
    hmm_model_path = os.path.join(MODEL_DIR, 'hmm_model.pkl')
    if os.path.exists(hmm_model_path):
        print("HMM 模型檔案已存在，無需重新訓練。")
        return joblib.load(hmm_model_path)

    print("開始訓練 HMM 模型...")
    hmm_observations_sequence = np.array([label_map[r] for r in all_history if r in ['B', 'P']]).reshape(-1, 1)

    if hmm_observations_sequence.size < 10: # HMM 需要足夠的序列數據
        print("HMM 訓練數據不足 (至少需要10個莊閒結果)，跳過 HMM 訓練。")
        return None
    
    # 檢查數據中是否至少有兩種不同的觀察值
    if len(np.unique(hmm_observations_sequence)) < 2:
        print("HMM 訓練數據過於單一 (沒有足夠的莊閒變化)，跳過 HMM 訓練。")
        return None

    # 設置 HMM 模型 (2 個隱藏狀態: 莊趨勢, 閒趨勢)
    # 這裡嘗試使用 GMMHMM，它對數據分佈的假設更具彈性
    # n_components=2 (2個隱藏狀態), n_mix=1 (每個狀態一個高斯分量), covariance_type="diag"
    hmm_model = hmm.GMMHMM(n_components=2, n_mix=1, covariance_type="diag", n_iter=200, random_state=42)
    
    try:
        # 增加重試機制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                hmm_model.fit(hmm_observations_sequence) 
                # 檢查 emissionprob_ 是否存在，只有成功訓練後才會有
                if hasattr(hmm_model, 'emissionprob_') and hmm_model.emissionprob_ is not None:
                    joblib.dump(hmm_model, hmm_model_path)
                    print(f"HMM 模型訓練完成並已儲存 (嘗試 {attempt + 1}/{max_retries})。")
                    return hmm_model
                else:
                    print(f"HMM 模型訓練後缺少 'emissionprob_' 屬性或為 None，訓練可能未成功 (嘗試 {attempt + 1}/{max_retries})。")
            except Exception as e:
                print(f"HMM 模型訓練失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("重試 HMM 模型訓練...")
        print("HMM 模型多次訓練失敗。")
        return None
    except Exception as e:
        print(f"HMM 模型訓練失敗: {e}")
        return None


def train_models_if_needed():
    """
    檢查模型檔案是否存在，若不存在則訓練並儲存模型。
    現在會先訓練 HMM，然後將 HMM 的輸出作為特徵訓練 XGBoost。
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
    if hmm_model is None:
        print("HMM 模型未能成功訓練或載入，XGBoost 將不包含 HMM 特徵。")
    
    # 2. 檢查 XGBoost 模型是否需要訓練
    model_files = ['xgb_model.pkl', 'scaler.pkl'] # 只檢查 XGBoost 和 scaler
    all_other_models_exist = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files)

    if not all_other_models_exist:
        print("偵測到 XGBoost 模型檔案不存在，正在自動執行訓練...")
        
        # 準備包含 HMM 特徵的訓練資料
        X_train, y_train = prepare_training_data(all_history, hmm_model)
        
        # 特徵數量現在是 5 (原始) + 2 (HMM機率) = 7
        expected_feature_dim = 7 
        if hmm_model is None or not hasattr(hmm_model, 'emissionprob_') or hmm_model.emissionprob_ is None: # 如果 HMM 未成功訓練，則預期特徵維度為 5
            expected_feature_dim = 5

        if X_train.shape[1] != expected_feature_dim:
             print(f"警告: 訓練數據特徵維度不符 (預期 {expected_feature_dim}, 實際 {X_train.shape[1]})。")
             print("這可能表示 HMM 訓練失敗或數據問題，將嘗試使用不含 HMM 特徵的數據訓練 XGBoost。")
             # 如果維度不符，且 HMM 未載入，則重新準備不含 HMM 特徵的訓練數據
             if hmm_model is None or not hasattr(hmm_model, 'emissionprob_') or hmm_model.emissionprob_ is None:
                 X_train, y_train = prepare_training_data(all_history, None) # 重新準備不含 HMM 特徵的數據
                 if X_train.shape[1] != 5: # 如果重新準備後仍不符，則返回
                     print("重新準備不含 HMM 特徵的數據後，特徵維度仍不符，無法訓練 XGBoost。")
                     return
             else:
                 print("訓練數據特徵維度錯誤，無法訓練 XGBoost。")
                 return

        if len(X_train) < 5 or len(np.unique(y_train)) < 2:
            print("訓練資料不足或不平衡，無法進行訓練。")
            return

        # 特徵縮放
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

        # 訓練與儲存 XGBoost 模型
        xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss',
                            learning_rate=0.1, max_depth=3, random_state=42)
        xgb.fit(X_scaled, y_train)
        joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgb_model.pkl'))

        print("XGBoost 模型訓練完成並已儲存。")
    else:
        print("XGBoost 模型檔案已存在，無需重新訓練。")


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
    current_history_from_file = load_data()
    
    if filtered_received_roadmap != current_history_from_file:
        save_data(filtered_received_roadmap)
        print(f"歷史數據已更新為：{filtered_received_roadmap}")
    else:
        print("收到的路紙與現有歷史數據一致，無需更新。")
    # --- 數據保存邏輯結束 ---
    
    # 實際用於特徵提取和預測的數據 (只考慮 'B'/'P'，因為模型只預測莊閒)
    current_game_sequence_for_prediction = [r for r in filtered_received_roadmap if r in ["B", "P"]]


    # 檢查所有模型檔案 (包括 XGBoost 和 scaler) 是否存在
    model_files = ['xgb_model.pkl', 'scaler.pkl']
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files):
        print("警告: 預測時發現 XGBoost 或 Scaler 模型檔案缺失，回傳預設機率。請檢查服務啟動日誌中的訓練過程。")
        return jsonify({
            "banker": 0.5,
            "player": 0.5,
            "tie": 0.05,
            "details": {
                "xgb": "N/A", "hmm": "N/A",
                "suggestion": "請先執行模型訓練任務。"
            }
        })
    
    # 載入所有已訓練好的模型和 Scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    
    hmm_model = None
    hmm_model_path = os.path.join(MODEL_DIR, 'hmm_model.pkl')
    if os.path.exists(hmm_model_path):
        hmm_model = joblib.load(hmm_model_path)

    # 特徵轉換與預測
    # 這裡使用 `current_game_sequence_for_prediction` 來提取特徵，並包含 HMM 特徵
    features, hmm_prediction = extract_features(current_game_sequence_for_prediction, hmm_model)
    
    # 這裡的 features.size 應該是 5 (原始) + 2 (HMM的2個機率) = 7
    expected_feature_size = 7 
    if hmm_model is None or not hasattr(hmm_model, 'emissionprob_') or hmm_model.emissionprob_ is None: # 如果 HMM 未成功載入或訓練，則預期特徵維度為 5
        expected_feature_size = 5

    if features.size != expected_feature_size: 
        print(f"警告: 特徵數量不符 (預期 {expected_feature_size}, 實際 {features.size})，回傳預設機率。")
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

    xgb_pred = reverse_map[np.argmax(xgb_pred_prob)]

    # 加權整合 (現在只有 XGBoost，所以直接使用其預測機率)
    banker_prob = xgb_pred_prob[0]
    player_prob = xgb_pred_prob[1]
    
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
        "banker": float(round(banker_prob, 3)), # 轉換為標準 Python float
        "player": float(round(player_prob, 3)), # 轉換為標準 Python float
        "tie": float(round(tie, 3)),           # 轉換為標準 Python float
        "details": {
            "xgb": f"{xgb_pred} ({float(xgb_pred_prob[np.argmax(xgb_pred_prob)]):.2f})", # 轉換為標準 Python float
            "hmm": hmm_prediction, # HMM 的獨立預測結果
            "suggestion": suggestion
        }
    })

# 原有的 `if __name__ == "__main__":` 區塊已簡化，僅用於本地開發測試
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
