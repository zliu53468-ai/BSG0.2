from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import NotFittedError

app = Flask(__name__)
CORS(app)

# === 初始化一個全局的、可持續學習的 AI 模型 ===
online_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
label_map = {'莊': 0, '閒': 1}
classes = np.array([0, 1])

def extract_features(roadmap):
    """
    特徵提取器：將牌路轉換為 AI 能理解的數字。
    """
    window = roadmap[-20:] # 分析最近20手
    b_count = window.count('莊')
    p_count = window.count('閒')
    total = b_count + p_count
    b_ratio = b_count / total if total > 0 else 0.5
    p_ratio = p_count / total if total > 0 else 0.5

    streak = 0
    last = window[-1] if window else None
    for item in reversed(window):
        if item == last:
            streak += 1
        else:
            break
    
    streak_type = 0 if last == '莊' else 1
    return np.array([b_ratio, p_ratio, streak, streak_type])

def hmm_predict(roadmap):
    """
    規則型 HMM 模擬器：專門判斷序列關係。
    """
    if len(roadmap) < 2: return "等待", 0.5
    last_two = roadmap[-2:]
    if last_two == ['莊', '閒']: return "閒", 0.65
    if last_two == ['閒', '莊']: return "莊", 0.65
    if last_two == ['莊', '莊']: return "莊", 0.7
    if last_two == ['閒', '閒']: return "閒", 0.7
    return "等待", 0.5

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "success", "message": "AI Engine is running."})

@app.route("/predict", methods=["POST"])
def predict():
    global online_model
    data = request.get_json()
    roadmap = data.get("roadmap", [])
    
    filtered_roadmap = [r for r in roadmap if r in ['莊', '閒']]

    if len(filtered_roadmap) < 5:
        return jsonify({
            "banker": 0.5, "player": 0.5, "tie": 0.05,
            "details": { "hmm": "資料不足" }
        })

    # --- 即時學習 ---
    if len(filtered_roadmap) > 1:
        unique_outcomes = set(filtered_roadmap)
        if len(unique_outcomes) > 1:
            X_train = extract_features(filtered_roadmap[:-1]).reshape(1, -1)
            y_train = np.array([label_map[filtered_roadmap[-1]]])
            online_model.partial_fit(X_train, y_train, classes=classes)

    # --- 預測 ---
    current_features = extract_features(filtered_roadmap).reshape(1, -1)
    
    try:
        sgd_pred_prob = online_model.predict_proba(current_features)[0]
    except NotFittedError:
        sgd_pred_prob = np.array([0.5, 0.5])

    hmm_pred, hmm_conf = hmm_predict(filtered_roadmap)
    hmm_prob = np.array([hmm_conf, 1 - hmm_conf]) if hmm_pred == '莊' else np.array([1 - hmm_conf, hmm_conf])

    # --- 融合預測 ---
    final_prob = (sgd_pred_prob * 0.7) + (hmm_prob * 0.3)
    
    banker_prob = final_prob[0]
    player_prob = final_prob[1]

    return jsonify({
        "banker": round(float(banker_prob), 3),
        "player": round(float(player_prob), 3),
        "tie": 0.05,
        "details": {
            "hmm": f"{hmm_pred} ({hmm_conf:.2f})"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
