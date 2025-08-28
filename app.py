from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

app = Flask(__name__)
CORS(app)

# === 1. PyTorch 神經網路定義 ===
class BaccaratNet(nn.Module):
    def __init__(self, input_size=5):
        super(BaccaratNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2) # 輸出莊、閒的原始分數
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# === 2. 初始化全局模型和優化器 ===
# 讓模型和其學習進度在所有請求之間共享
input_feature_size = 5
pt_model = BaccaratNet(input_size=input_feature_size)
optimizer = optim.Adam(pt_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
model_is_ready = False # 標記模型是否已進行過初始學習

label_map = {'莊': 0, '閒': 1}

# === 3. 特徵提取器 ===
def extract_features(roadmap):
    window = roadmap[-20:]
    b_count = window.count('莊')
    p_count = window.count('閒')
    total = b_count + p_count
    b_ratio = b_count / total if total > 0 else 0.5
    p_ratio = p_count / total if total > 0 else 0.5

    streak = 0
    last = window[-1] if window else None
    for item in reversed(window):
        if item == last: streak += 1
        else: break
    
    streak_type = 0 if last == '莊' else 1
    
    # 新增單跳特徵
    jumps = sum(1 for i in range(len(window) - 1) if window[i] != window[i+1])
    jump_ratio = jumps / len(window) if len(window) > 0 else 0.0

    return np.array([b_ratio, p_ratio, streak, streak_type, jump_ratio])

# === 4. 規則型 HMM 模擬器 ===
def hmm_predict(roadmap):
    if len(roadmap) < 2: return "等待", 0.5
    last_two = roadmap[-2:]
    if last_two == ['莊', '閒']: return "閒", 0.65
    if last_two == ['閒', '莊']: return "莊", 0.65
    if last_two == ['莊', '莊']: return "莊", 0.7
    if last_two == ['閒', '閒']: return "閒", 0.7
    return "等待", 0.5

# === 5. API 端點 ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "success", "message": "PyTorch AI Engine is running."})

@app.route("/predict", methods=["POST"])
def predict():
    global pt_model, optimizer, model_is_ready
    data = request.get_json()
    roadmap = data.get("roadmap", [])
    
    filtered_roadmap = [r for r in roadmap if r in ['莊', '閒']]

    if len(filtered_roadmap) < 5:
        return jsonify({
            "banker": 0.5, "player": 0.5, "tie": 0.05,
            "details": {"pytorch": "資料不足", "hmm": "資料不足"}
        })

    # --- 即時學習 ---
    if len(filtered_roadmap) > 1:
        pt_model.train() # 切換到訓練模式
        X_train_np = extract_features(filtered_roadmap[:-1])
        y_train_np = label_map[filtered_roadmap[-1]]
        
        X_train = torch.FloatTensor(X_train_np).unsqueeze(0)
        y_train = torch.LongTensor([y_train_np])

        # 進行幾次微調
        for _ in range(5):
            optimizer.zero_grad()
            outputs = pt_model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        model_is_ready = True

    # --- 預測 ---
    pt_model.eval() # 切換到評估模式
    current_features_np = extract_features(filtered_roadmap)
    current_features = torch.FloatTensor(current_features_np).unsqueeze(0)
    
    with torch.no_grad():
        if model_is_ready:
            outputs = pt_model(current_features)
            pt_pred_prob = torch.softmax(outputs, dim=1).squeeze().numpy()
        else:
            pt_pred_prob = np.array([0.5, 0.5])

    hmm_pred, hmm_conf = hmm_predict(filtered_roadmap)
    hmm_prob = np.array([hmm_conf, 1 - hmm_conf]) if hmm_pred == '莊' else np.array([1 - hmm_conf, hmm_conf])

    # --- 融合預測 (PyTorch 權重 70%, HMM 權重 30%) ---
    final_prob = (pt_pred_prob * 0.7) + (hmm_prob * 0.3)
    
    banker_prob = final_prob[0]
    player_prob = final_prob[1]

    pt_pred_label = "莊" if np.argmax(pt_pred_prob) == 0 else "閒"

    return jsonify({
        "banker": round(float(banker_prob), 3),
        "player": round(float(player_prob), 3),
        "tie": 0.05,
        "details": {
            "pytorch": f"{pt_pred_label} ({max(pt_pred_prob):.2f})",
            "hmm": f"{hmm_pred} ({hmm_conf:.2f})",
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
