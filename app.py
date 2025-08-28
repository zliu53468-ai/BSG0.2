from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- PyTorch 神經網路定義 ---
class Net(nn.Module):
    def __init__(self, input_size=5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2) # 輸出兩個類別：莊、閒
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

app = Flask(__name__)
CORS(app)

X_data = []
y_data = []

label_map = {'莊': 0, '閒': 1}
reverse_map = {0: '莊', 1: '閒'}

def extract_features(roadmap):
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
    return np.array([b_ratio, p_ratio, streak, streak_type, prev])

def prepare_training_data(roadmap):
    filtered = [r for r in roadmap if r in ['莊', '閒']]
    X = []
    y = []
    for i in range(1, len(filtered)):
        features = extract_features(filtered[:i])
        X.append(features)
        y.append(label_map[filtered[i]])
    return np.array(X), np.array(y)

def hmm_predict(roadmap):
    filtered = [r for r in roadmap if r in ['莊', '閒']]
    if len(filtered) < 2:
        return "等待", 0.5
    last_two = filtered[-2:]
    if last_two == ['莊', '閒']: return "閒", 0.6
    if last_two == ['閒', '莊']: return "莊", 0.6
    if last_two == ['莊', '莊']: return "莊", 0.7
    if last_two == ['閒', '閒']: return "閒", 0.7
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
    if len(last_valid) < 5:
        return jsonify({
            "banker": 0.5, "player": 0.5, "tie": 0.05,
            "details": {
                "sgd": "資料不足", "xgb": "資料不足",
                "lgb": "資料不足", "hmm": "資料不足",
                "pytorch": "資料不足", "suggestion": "請多輸入路紙"
            }
        })

    X_new, y_new = prepare_training_data(roadmap)
    global X_data, y_data
    if len(X_new) > 0:
        X_data = X_new
        y_data = y_new

    features = extract_features(roadmap).reshape(1, -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    features_scaled = scaler.transform(features)

    # --- 模型訓練與預測 ---
    # SGD
    sgd = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    sgd.fit(X_scaled, y_data)
    sgd_pred_prob = sgd.predict_proba(features_scaled)[0]
    sgd_pred = reverse_map[np.argmax(sgd_pred_prob)]

    # XGBoost
    xgb = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_scaled, y_data)
    xgb_pred_prob = xgb.predict_proba(features_scaled)[0]
    xgb_pred = reverse_map[np.argmax(xgb_pred_prob)]

    # LightGBM
    lgbm = LGBMClassifier(n_estimators=50)
    lgbm.fit(X_scaled, y_data)
    lgb_pred_prob = lgbm.predict_proba(features_scaled)[0]
    lgb_pred = reverse_map[np.argmax(lgb_pred_prob)]

    # PyTorch
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y_data)
    features_tensor = torch.FloatTensor(features_scaled)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    pt_model = Net(input_size=X_scaled.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.01)
    
    # 訓練 PyTorch 模型
    for epoch in range(20): # 訓練 20 次
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = pt_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    pt_model.eval()
    with torch.no_grad():
        pt_pred_prob = pt_model(features_tensor).numpy()[0]
    pt_pred = reverse_map[np.argmax(pt_pred_prob)]

    # HMM
    hmm_pred, hmm_prob = hmm_predict(roadmap)

    # --- 綜合結果 ---
    banker = np.mean([sgd_pred_prob[0], xgb_pred_prob[0], lgb_pred_prob[0], pt_pred_prob[0], hmm_prob if hmm_pred == "莊" else 1-hmm_prob])
    player = np.mean([sgd_pred_prob[1], xgb_pred_prob[1], lgb_pred_prob[1], pt_pred_prob[1], hmm_prob if hmm_pred == "閒" else 1-hmm_prob])
    tie = 0.05

    suggestion = "等待"
    if banker > player and banker > tie: suggestion = "莊"
    elif player > banker and player > tie: suggestion = "閒"

    return jsonify({
        "banker": round(banker, 3),
        "player": round(player, 3),
        "tie": round(tie, 3),
        "details": {
            "sgd": f"{sgd_pred} ({sgd_pred_prob[np.argmax(sgd_pred_prob)]:.2f})",
            "xgb": f"{xgb_pred} ({xgb_pred_prob[np.argmax(xgb_pred_prob)]:.2f})",
            "lgb": f"{lgb_pred} ({lgb_pred_prob[np.argmax(lgb_pred_prob)]:.2f})",
            "pytorch": f"{pt_pred} ({pt_pred_prob[np.argmax(pt_pred_prob)]:.2f})",
            "hmm": f"{hmm_pred} ({hmm_prob:.2f})",
            "suggestion": suggestion
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
