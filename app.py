from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import NotFittedError

app = Flask(__name__)
CORS(app)

# === 1. 初始化全局的、可持續學習的 AI 模型 (Deep Learning Prediction AI) ===
online_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
label_map = {'莊': 0, '閒': 1}
classes = np.array([0, 1])

# === 2. 特徵提取器 (供學習模型使用) ===
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
    return np.array([b_ratio, p_ratio, streak, streak_type])

# === 3. 俚語與牌路分析模組 (Statistical Pattern Recognition & Big Road Analysis) ===
def analyze_big_road_patterns(roadmap):
    if not roadmap: return None, "等待"
    
    # 長龍判斷
    if len(roadmap) >= 4 and len(set(roadmap[-4:])) == 1:
        return np.array([0.7, 0.3]) if roadmap[-1] == '莊' else np.array([0.3, 0.7]), f"長{roadmap[-1]}"
        
    # 單跳判斷
    if len(roadmap) >= 4 and roadmap[-4:] == ['莊', '閒', '莊', '閒']:
        return np.array([0.65, 0.35]), "單跳" # 預測下個是莊
    if len(roadmap) >= 4 and roadmap[-4:] == ['閒', '莊', '閒', '莊']:
        return np.array([0.35, 0.65]), "單跳" # 預測下個是閒

    return None, "常規"

# === 4. 下三路分析模擬器 (Down Three Roads Analysis) ===
def analyze_down_three_roads(roadmap):
    # 這是一個簡化的模擬器，用來模擬下三路的邏輯
    # 真實的下三路需要複雜的 grid 轉換，但我們可以模擬其核心思想：尋找規律性
    if len(roadmap) < 10: return "資料不足", "資料不足", "資料不足"
    
    # 大眼仔路: 模擬整齊度
    big_eye = "莊" if roadmap.count('莊') % 2 == 0 else "閒"
    # 小路: 模擬有無對應
    small_road = "莊" if (roadmap[-1] == roadmap[-3]) else "閒"
    # 甲由路: 模擬直落
    cockroach_road = "莊" if (roadmap[-1] == roadmap[-4]) else "閒"
    
    return big_eye, small_road, cockroach_road

# === 5. API 端點 (Multi-Source Data Fusion AI) ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "success", "message": "Multi-Model AI Engine is running."})

@app.route("/predict", methods=["POST"])
def predict():
    global online_model
    data = request.get_json()
    roadmap = data.get("roadmap", [])
    
    filtered_roadmap = [r for r in roadmap if r in ['莊', '閒']]

    if len(filtered_roadmap) < 5:
        return jsonify({
            "banker": 0.5, "player": 0.5, "tie": 0.05,
            "details": { "analysis": "資料不足，請多輸入路紙" }
        })

    # --- 各模型獨立分析 ---
    # 1. 統計分析
    b_count = filtered_roadmap.count('莊')
    p_count = filtered_roadmap.count('閒')
    total = b_count + p_count
    stats_prob = np.array([b_count / total, p_count / total]) if total > 0 else np.array([0.5, 0.5])

    # 2. 俚語/牌路分析
    big_road_prob, big_road_text = analyze_big_road_patterns(filtered_roadmap)

    # 3. 下三路分析
    big_eye, small_road, cockroach_road = analyze_down_three_roads(filtered_roadmap)
    d3r_b_score = (1 if big_eye == '莊' else 0) + (1 if small_road == '莊' else 0) + (1 if cockroach_road == '莊' else 0)
    d3r_p_score = 3 - d3r_b_score
    d3r_prob = np.array([d3r_b_score / 3, d3r_p_score / 3])

    # 4. 深度學習模型 (即時學習與預測)
    if len(filtered_roadmap) > 1 and len(set(filtered_roadmap)) > 1:
        X_train = extract_features(filtered_roadmap[:-1]).reshape(1, -1)
        y_train = np.array([label_map[filtered_roadmap[-1]]])
        online_model.partial_fit(X_train, y_train, classes=classes)
    
    try:
        current_features = extract_features(filtered_roadmap).reshape(1, -1)
        learning_prob = online_model.predict_proba(current_features)[0]
    except NotFittedError:
        learning_prob = np.array([0.5, 0.5])

    # --- 智慧融合 (Multi-Source Fusion) ---
    # 如果有明顯的俚語牌路，給予最高權重
    if big_road_prob is not None:
        final_prob = (learning_prob * 0.3) + (big_road_prob * 0.5) + (d3r_prob * 0.2)
    else: # 否則，讓學習模型和下三路佔主導
        final_prob = (learning_prob * 0.5) + (d3r_prob * 0.3) + (stats_prob * 0.2)
    
    banker_prob = final_prob[0]
    player_prob = final_prob[1]

    # --- 準備回傳的詳細分析文字 ---
    d3r_text = f"大眼仔路: {big_eye}, 小路: {small_road}, 甲由路: {cockroach_road}"
    stats_text = f"莊出現率 {(stats_prob[0]*100):.1f}%, 閒出現率 {(stats_prob[1]*100):.1f}%"
    suggestion = "莊" if banker_prob > player_prob else "閒"

    return jsonify({
        "banker": round(float(banker_prob), 3),
        "player": round(float(player_prob), 3),
        "tie": 0.05,
        "details": {
            "deep_learning_prediction": f"{'莊' if np.argmax(learning_prob) == 0 else '閒'}",
            "statistical_pattern_recognition": suggestion,
            "big_road_analysis": big_road_text,
            "down_three_roads_analysis": d3r_text,
            "statistics": stats_text
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
