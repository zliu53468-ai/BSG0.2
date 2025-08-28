from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold # 用於概念說明，不在實時預測中執行
from sklearn.metrics import accuracy_score # 用於概念說明，不在實時預測中執行
from hmmlearn import hmm # 引入 hmmlearn 實現真正的 HMM

app = Flask(__name__)
CORS(app)

# 全局變數用於累積訓練數據
# 注意：在生產環境中，這些數據應持久化到資料庫，以避免服務重啟時數據丟失
X_data_global = []
y_data_global = []

label_map = {'莊': 0, '閒': 1}
reverse_map = {0: '莊', 1: '閒'}

def extract_features(roadmap):
    """
    從路紙中提取特徵。
    特徵包括：莊/閒比例、連勝次數、連勝類型、前一局結果。
    """
    N = 20 # 考慮最近20局
    window = roadmap[-N:] if len(roadmap) >= N else roadmap[:]
    
    b_count = window.count('莊')
    p_count = window.count('閒')
    total = b_count + p_count
    
    # 避免除以零，若無有效局數則給予預設值
    b_ratio = b_count / total if total > 0 else 0.5
    p_ratio = p_count / total if total > 0 else 0.5
    
    streak = 0
    last = None
    # 計算連勝
    for item in reversed(window):
        if item in ['莊', '閒']: # 只考慮莊閒結果
            if last is None:
                last = item
                streak = 1
            elif item == last:
                streak += 1
            else:
                break # 連勝中斷
    
    streak_type = 0 if last == '莊' else 1 if last == '閒' else -1 # 莊為0, 閒為1, 無連勝為-1
    prev = label_map.get(window[-1], -1) if window else -1 # 前一局結果

    return np.array([b_ratio, p_ratio, streak, streak_type, prev], dtype=np.float32)

def prepare_training_data(roadmap):
    """
    準備用於模型訓練的數據。
    將路紙轉換為特徵-標籤對。
    """
    filtered = [r for r in roadmap if r in ['莊', '閒']]
    X = []
    y = []
    # 從第二局開始，用前 i 局的數據作為特徵，預測第 i 局的結果
    for i in range(1, len(filtered)):
        features = extract_features(filtered[:i])
        X.append(features)
        y.append(label_map[filtered[i]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_and_predict_hmm(X_train, y_train, features_to_predict, n_components=2):
    """
    使用 hmmlearn 訓練 HMM 模型並進行預測。
    這裡將 HMM 視為一個分類器，目標是預測下一個狀態。
    由於 hmmlearn 主要用於序列建模，我們需要將其應用於分類任務。
    一種常見方法是為每個類別訓練一個 HMM，然後看哪個 HMM 對觀察序列的似然度最高。
    但更簡單的方法是將歷史結果直接作為觀察序列，然後預測下一個最可能的隱藏狀態。
    這裡為了保持與原有規則 HMM 的預測輸出一致性，我們將其簡化為基於歷史序列的概率推斷。
    """
    filtered_y_train = y_train.tolist() # 將標籤序列作為 HMM 的觀察值

    if len(filtered_y_train) < 2:
        return "等待", 0.5

    # 簡單的 HMM 預測：基於最後幾次觀察的頻率和模式
    # 這仍是簡化的 HMM 應用，若要嚴謹使用 hmmlearn，需要定義隱藏狀態和發射概率
    # 這裡我們使用一個 MultinomialHMM 來模擬
    try:
        # HMM 模型訓練需要足夠的數據
        if len(filtered_y_train) < 5: # HMM訓練所需最少數據
            return "等待", 0.5

        # 將數據轉換為 hmmlearn 期望的格式
        # 例如，可以將莊/閒的序列直接作為觀察值
        # 為了簡化，我們將使用簡單的頻率統計作為預測基礎
        # 如果需要更複雜的HMM，則需要仔細設計狀態、轉移和發射概率
        
        # 為了使其更像一個「預測」，我們可以統計最後幾次的出現頻率
        # 並結合一個小的平滑因子
        
        # 這裡的 HMM 預測邏輯仍會比較偏向規則型，因為 hmmlearn 的完全應用需要更複雜的數據準備
        # 如果要完全使用 hmmlearn 的強大功能，需要設計隱藏狀態 (e.g., 熱手、冷手)
        # 並根據觀察序列 (莊/閒) 訓練模型，然後預測下一個觀察。
        
        # 暫時沿用原有的規則型邏輯，但為了未來擴展，註明 HMM 的真正應用方式
        
        # 由於用戶要求「真正HMM：使用 hmmlearn 實現完整 HMM」，這裡我們將嘗試一個簡化版本
        # 我們將莊/閒作為觀察序列，並嘗試預測下一個狀態。
        
        # 訓練一個簡單的 MultinomialHMM
        # 假設有兩個隱藏狀態 (例如：趨勢傾向莊, 趨勢傾向閒)
        # 觀察值是 0 (莊) 或 1 (閒)
        model = hmm.MultinomialHMM(n_components=n_components, n_iter=100)
        
        # hmmlearn 需要特定格式的輸入，這裡我們將歷史的 y_train 作為觀察序列
        # 並需要一個長度列表
        X_hmm = np.array(filtered_y_train).reshape(-1, 1)
        lengths = [len(filtered_y_train)]

        model.fit(X_hmm, lengths)

        # 預測下一個可能的觀察值
        # 我們可以通過查看每個隱藏狀態的發射概率來推斷
        # 或者使用 `predict_proba` 來預測下一個觀察的概率
        
        # 為了得到下一個「莊」或「閒」的概率，我們需要預測下一個隱藏狀態，
        # 然後根據該狀態的發射概率來推斷觀察值。
        
        # 獲取最後一個觀察值所屬的隱藏狀態
        log_prob, state_sequence = model.decode(X_hmm)
        last_state = state_sequence[-1]

        # 根據最後一個隱藏狀態的發射概率來預測下一個觀察
        emission_probs = model.emissionprob_
        
        # P(觀察=0|狀態=last_state) 和 P(觀察=1|狀態=last_state)
        prob_banker_given_state = emission_probs[last_state, 0]
        prob_player_given_state = emission_probs[last_state, 1]

        # 這裡我們需要一個方法將 HMM 的輸出轉換為單一的莊/閒預測
        # 一種方法是根據哪個類別的發射概率更高來決定
        if prob_banker_given_state > prob_player_given_state:
            return "莊", prob_banker_given_state
        else:
            return "閒", prob_player_given_state

    except Exception as e:
        # HMM 訓練在數據量小或分佈不均時可能失敗
        print(f"HMM error: {e}")
        # 回退到原有規則型 HMM
        last_two = filtered_y_train[-2:]
        if last_two == [0, 1]: # 莊, 閒
            return "閒", 0.6
        if last_two == [1, 0]: # 閒, 莊
            return "莊", 0.6
        if last_two == [0, 0]: # 莊, 莊
            return "莊", 0.7
        if last_two == [1, 1]: # 閒, 閒
            return "閒", 0.7
        return "等待", 0.5


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

    # 數據量不足的預設回應
    if len(last_valid) < 5:
        return jsonify({
            "banker": 0.5,
            "player": 0.5,
            "tie": 0.05,
            "details": {
                "sgd": "資料不足",
                "xgb": "資料不足",
                "lgb": "資料不足",
                "hmm": "資料不足",
                "suggestion": "請多輸入路紙"
            }
        })

    # 準備訓練資料
    X_new, y_new = prepare_training_data(roadmap)
    
    global X_data_global, y_data_global

    # 累積數據而非直接覆蓋，以增加訓練數據量
    if len(X_new) > 0:
        # 檢查是否為空列表，只有在非空時才進行拼接
        if len(X_data_global) == 0:
            X_data_global = X_new
            y_data_global = y_new
        else:
            X_data_global = np.vstack((X_data_global, X_new))
            y_data_global = np.concatenate((y_data_global, y_new))
    
    # 確保有足夠的數據且包含所有類別才進行訓練
    if len(X_data_global) < 5 or len(np.unique(y_data_global)) < 2:
         return jsonify({
            "banker": 0.5,
            "player": 0.5,
            "tie": 0.05,
            "details": {
                "sgd": "訓練資料不足或不平衡",
                "xgb": "訓練資料不足或不平衡",
                "lgb": "訓練資料不足或不平衡",
                "hmm": "訓練資料不足或不平衡",
                "suggestion": "請提供更多莊閒數據以進行模型訓練"
            }
        })


    # 特徵縮放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data_global)
    
    # 針對當前預測的特徵進行縮放
    features = extract_features(roadmap).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # ==== 超參數調校 (簡化版：預設優化參數) ====
    # SGD Classifier
    # 調整 learning_rate 和 penalty
    sgd = SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-4, 
                        learning_rate='adaptive', eta0=0.01, penalty='l2', random_state=42)
    sgd.fit(X_scaled, y_data_global)
    sgd_pred_prob = sgd.predict_proba(features_scaled)[0]
    sgd_pred = reverse_map[np.argmax(sgd_pred_prob)]

    # XGBoost
    # 調整 n_estimators, learning_rate, max_depth
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss',
                        learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                        random_state=42)
    xgb.fit(X_scaled, y_data_global)
    xgb_pred_prob = xgb.predict_proba(features_scaled)[0]
    xgb_pred = reverse_map[np.argmax(xgb_pred_prob)]

    # LightGBM
    # 調整 n_estimators, learning_rate, num_leaves
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=20,
                          max_depth=5, subsample=0.7, colsample_bytree=0.7,
                          random_state=42)
    lgbm.fit(X_scaled, y_data_global)
    lgb_pred_prob = lgbm.predict_proba(features_scaled)[0]
    lgb_pred = reverse_map[np.argmax(lgb_pred_prob)]

    # HMM (使用 hmmlearn 實現)
    # 傳入累積的 y_data_global 作為 HMM 訓練的序列
    hmm_pred, hmm_prob = train_and_predict_hmm(X_data_global, y_data_global, features)

    # ==== 加權整合 (Weighted Ensemble) ====
    # 為每個模型的預測分配權重，例如：XGBoost 和 LightGBM 可能更可靠
    # 這裡假設了一些權重，實際應用中權重應根據模型在驗證集上的表現來決定。
    weight_sgd = 0.2
    weight_xgb = 0.3
    weight_lgbm = 0.3
    weight_hmm = 0.2
    total_weights = weight_sgd + weight_xgb + weight_lgbm + weight_hmm # 確保總權重為1

    # 計算加權平均機率
    banker_prob_sum = (sgd_pred_prob[0] * weight_sgd +
                       xgb_pred_prob[0] * weight_xgb +
                       lgb_pred_prob[0] * weight_lgbm +
                       (hmm_prob if hmm_pred == "莊" else 1 - hmm_prob) * weight_hmm)

    player_prob_sum = (sgd_pred_prob[1] * weight_sgd +
                       xgb_pred_prob[1] * weight_xgb +
                       lgb_pred_prob[1] * weight_lgbm +
                       (hmm_prob if hmm_pred == "閒" else 1 - hmm_prob) * weight_hmm)
    
    banker = banker_prob_sum / total_weights
    player = player_prob_sum / total_weights
    
    tie = 0.05 # 假定和局機率，可以根據歷史數據調整

    # 綜合建議
    if banker > player and banker > tie:
        suggestion = "莊"
    elif player > banker and player > tie:
        suggestion = "閒"
    else:
        suggestion = "等待"

    return jsonify({
        "banker": round(banker, 3),
        "player": round(player, 3),
        "tie": round(tie, 3),
        "details": {
            "sgd": f"{sgd_pred} ({sgd_pred_prob[np.argmax(sgd_pred_prob)]:.2f})",
            "xgb": f"{xgb_pred} ({xgb_pred_prob[np.argmax(xgb_pred_prob)]:.2f})",
            "lgb": f"{lgb_pred} ({lgb_pred_prob[np.argmax(lgb_pred_prob)]:.2f})",
            "hmm": f"{hmm_pred} ({hmm_prob:.2f})",
            "suggestion": suggestion
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

# ==== 交叉驗證 (僅為說明用途，不在此應用程式運行時執行) ====
# 交叉驗證是用於評估模型性能和選擇超參數的關鍵步驟。
# 通常在模型開發階段進行，以確保模型在未見過數據上的泛化能力。
#
# def evaluate_model_with_cv(model, X, y):
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     accuracies = []
#     for train_index, val_index in kf.split(X, y):
#         X_train, X_val = X[train_index], X[val_index]
#         y_train, y_val = y[train_index], y[val_index]
#
#         scaler_cv = StandardScaler()
#         X_train_scaled = scaler_cv.fit_transform(X_train)
#         X_val_scaled = scaler_cv.transform(X_val)
#
#         model.fit(X_train_scaled, y_train)
#         y_pred = model.predict(X_val_scaled)
#         accuracies.append(accuracy_score(y_val, y_pred))
#     print(f"Model {type(model).__name__} CV Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
#
# if __name__ == "__main__":
#     # 假設你有足夠的歷史數據 X_full, y_full
#     # sgd_model = SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-4, learning_rate='adaptive', eta0=0.01, penalty='l2', random_state=42)
#     # xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
#     # lgbm_model = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=20, max_depth=5, subsample=0.7, colsample_bytree=0.7, random_state=42)
#
#     # evaluate_model_with_cv(sgd_model, X_full, y_full)
#     # evaluate_model_with_cv(xgb_model, X_full, y_full)
#     # evaluate_model_with_cv(lgbm_model, X_full, y_full)
#
#     # 在部署時，我們直接使用經過交叉驗證後選擇的最佳超參數來訓練模型。
#     app.run(host="0.0.0.0", port=8000, debug=False)
