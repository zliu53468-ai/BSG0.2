# -*- coding: utf-8 -*-
import os
import json
import logging
import time
import re
from datetime import datetime, timedelta
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
# 全域變數與模型預載
# =============================================================================
MODEL_DIR = 'models'
DATA_FILE = 'data/history_data.json'
LABEL_MAP = {'B': 0, 'P': 1, 'T': 2}
REVERSE_MAP = {0: 'B', 1: 'P', 2: 'T'}
CHINESE_MAP = {'庄': 'B', '莊': 'B', '闲': 'P', '閒': 'P', '和': 'T'}
N_FEATURES_WINDOW = 20
models = {}
models_loaded = False

# 用户会话管理
user_sessions = {}
user_daily_usage = {}

# 确保数据目录存在
if not os.path.exists('data'):
    os.makedirs('data')

# 预先喂养的数据
PRELOADED_DATA = [
    "P", "P", "T", "B", "T", "B", "P", "B", "P", "P", "B", "B", "T", "B", "B", "P", "B", "B", "P", "B", 
    "B", "T", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "P", 
    "B", "P", "B", "P", "T", "P", "B", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "B", "P", "B", 
    "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "T", 
    "B", "P", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "P", "B", "P", "T", "T", "B", "P",
    "P", "P", "B", "B", "B", "B", "T", "T", "T", "B", "B", "B", "B", "B", "B", "P", "P", "P", "T", "P", 
    "T", "B", "P", "P", "T", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "P", "B", 
    "B", "P", "B", "B", "B", "B", "P", "P", "P", "P", "P", "T", "P", "B", "P", "P", "B", "T", "B", "B", 
    "B", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "P", "B", "P", "P", "B", "P", "B", "B", 
    "P", "B", "P", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", 
    "T", "T", "B", "P", "B", "B", "B", "T", "T", "B", "B", "P", "B", "T", "P", "B", "P", "B", "P", "P", 
    "P", "B", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", 
    "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", 
    "B", "B", "P", "B", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", 
    "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", "B", "B", 
    "P", "B", "B", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", 
    "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", "B", "B", "P", 
    "B", "B", "B", "B", "B", "B", "P", "B", "T", "T", "P", "B", "B", "B", "P", "B", "B", "P", "B", "P", 
    "B", "P", "B", "P", "P", "P", "P", "P", "P", "P", "B", "B", "B", "P", "T", "P", "B", "T", "B", "B", 
    "B", "B", "T", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "B", "B", "P", "P", "B", "P", 
    "P", "P", "P", "P", "B", "B", "B", "B", "B", "T", "B", "B", "P", "B", "P", "T", "P", "B", "P", "B", 
    "B", "P", "B", "B", "B", "P", "P", "P", "B", "P", "P", "B", "P", "P", "B", "B", "P", "P", "B", "P", 
    "B", "B", "B", "B", "B", "B", "B", "B", "P", "T", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", 
    "P", "P", "T", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "P", "B", "T", "P", "B", "B", 
    "P", "B", "P", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "T", "P", 
    "P", "B", "T", "B", "P", "P", "P", "B", "B", "P", "B", "B", "B", "P", "B", "P", "P", "B", "B", "B", 
    "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", 
    "B", "P", "P", "T", "P", "P", "P", "B", "P", "P", "P", "B", "B", "B", "P", "P", "B", "P", "B", "B", 
    "T", "P", "B", "P", "P", "T", "P", "P", "P", "B", "B", "P", "P", "T", "P", "T", "B", "T", "P", "B", 
    "P", "P", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "P", "T", "P", "P", "P", "B", "B", "P", 
    "P", "B", "P", "B", "P", "B", "B", "P", "T", "B", "P", "T", "T", "P", "T", "B", "T", "P", "T", "P", 
    "T", "P", "P", "B", "B", "P", "P", "P", "P", "P"
]

def load_all_models():
    global models, models_loaded
    if models_loaded: 
        return
    try:
        app.logger.info("⏳ 首次請求，開始載入 AI 專家模型...")
        models['scaler'] = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        models['xgb'] = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
        models['hmm'] = joblib.load(os.path.join(MODEL_DIR, 'hmm_model.pkl'))
        models['lgbm'] = joblib.load(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))
        models_loaded = True
        app.logger.info("✅ 所有 AI 專家模型已成功載入記憶體。")
    except Exception as e:
        app.logger.error(f"❌ 載入模型失敗: {e}", exc_info=True)
        models_loaded = False

def load_history_data():
    """加载历史数据，包括预先喂养的数据和用户新增数据"""
    # 尝试加载用户新增数据
    user_data = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                user_data = json.load(f)
        except:
            user_data = []
    
    # 合并预先喂养的数据和用户数据
    return PRELOADED_DATA + user_data

def save_history_data(new_data):
    """保存新的历史数据"""
    # 加载现有用户数据
    existing_data = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                existing_data = json.load(f)
        except:
            existing_data = []
    
    # 添加新数据
    existing_data.extend(new_data)
    
    # 保存数据
    with open(DATA_FILE, 'w') as f:
        json.dump(existing_data, f)
    
    return len(existing_data)

# =============================================================================
# 路單分析核心 (BaccaratAnalyzer)
# =============================================================================
class BaccaratAnalyzer:
    def __init__(self, roadmap):
        self.roadmap = roadmap
        self.big_road_grid = self._generate_big_road_grid()

    def _generate_big_road_grid(self):
        grid = []
        if not self.roadmap: return grid
        current_col, last_result = [], None
        for result in self.roadmap:
            if result != last_result and last_result is not None:
                grid.append(current_col)
                current_col = []
            current_col.append(result)
            last_result = result
        if current_col: grid.append(current_col)
        return grid

    def _get_col_len(self, c):
        return len(self.big_road_grid[c]) if 0 <= c < len(self.big_road_grid) else 0

    def _get_derived_bead_color(self, c, r, offset):
        if c < offset: return None
        if r == 0:
            return 'R' if self._get_col_len(c - 1) == self._get_col_len(c - offset -1) else 'B'
        ref_bead_exists = r < self._get_col_len(c - offset)
        ref_bead_above_exists = (r - 1) < self._get_col_len(c - offset)
        if ref_bead_exists: return 'R'
        elif ref_bead_above_exists: return 'B'
        return 'B'

    def get_derived_roads_data(self):
        roads = {'big_eye': [], 'small': [], 'cockroach': []}
        offsets = {'big_eye': 1, 'small': 2, 'cockroach': 3}
        for name, offset in offsets.items():
            derived_road_flat = []
            if len(self.big_road_grid) < offset + 1: continue
            for c in range(offset, len(self.big_road_grid)):
                start_row = 1 if len(self.big_road_grid[c - 1]) == 1 else 0
                for r in range(start_row, 6):
                    if r >= self._get_col_len(c): break
                    color = self._get_derived_bead_color(c, r, offset)
                    if color: derived_road_flat.append(color)
            if derived_road_flat:
                grid, current_col, last_bead = [], [], None
                for bead in derived_road_flat:
                    if bead != last_bead and last_bead is not None:
                        grid.append(current_col)
                        current_col = []
                    current_col.append(bead)
                    last_bead = bead
                if current_col: grid.append(current_col)
                roads[name] = grid
        return roads

    def get_derived_road_features(self):
        roads_data = self.get_derived_roads_data()
        features = []
        for name in ['big_eye', 'small', 'cockroach']:
            road = roads_data.get(name, [])
            flat_road = [bead for col in road for bead in col] if road else []
            if len(flat_road) > 5:
                window = flat_road[-10:]
                red_count = window.count('R')
                features.append(red_count / len(window))
            else:
                features.append(0.5)
        return features

# =============================================================================
# 預測函式
# =============================================================================
def get_hmm_prediction(hmm_model, roadmap_numeric):
    try:
        if len(roadmap_numeric) < 10:
            return "數據不足", 0.5
            
        hidden_states = hmm_model.predict(roadmap_numeric)
        last_state = hidden_states[-1]
        transition_probs = hmm_model.transmat_[last_state, :]
        emission_probs = hmm_model.emissionprob_
        prob_b = np.dot(transition_probs, emission_probs[:, 0])
        prob_p = np.dot(transition_probs, emission_probs[:, 1])
        total_prob = prob_b + prob_p
        
        if total_prob < 1e-9: 
            return "觀望", 0.5
            
        prob_b /= total_prob
        prob_p /= total_prob
        confidence = max(prob_b, prob_p)
        
        if abs(prob_b - prob_p) < 0.05:
            return "觀望", confidence
            
        return ("B" if prob_b > prob_p else "P"), confidence
    except Exception as e:
        app.logger.error(f"HMM預測錯誤: {e}")
        return "觀望", 0.5

def get_ml_prediction(model, scaler, roadmap):
    if len(roadmap) < N_FEATURES_WINDOW:
        return "數據不足", 0.5, 0.5, 0.5
        
    # 使用歷史數據創建分析器
    analyzer = BaccaratAnalyzer(roadmap)
    
    # 使用最後N個結果作為窗口
    window = roadmap[-N_FEATURES_WINDOW:]
    
    # 計算基本特徵
    b_count = window.count('B')
    p_count = window.count('P')
    t_count = window.count('T')
    total = b_count + p_count
    
    b_ratio = b_count / total if total > 0 else 0.5
    p_ratio = p_count / total if total > 0 else 0.5
    
    # 計算連續出現次數
    streak = 0
    last_result = None
    for item in reversed(window):
        if item in ['B', 'P']:
            if last_result is None:
                last_result = item
                streak = 1
            elif item == last_result:
                streak += 1
            else:
                break
                
    streak_type = LABEL_MAP.get(last_result, -1)
    prev_result = LABEL_MAP.get(window[-1], -1) if window else -1
    
    # 添加更多特徵
    short_window = roadmap[-5:] if len(roadmap) >= 5 else roadmap
    short_b_count = short_window.count('B')
    short_p_count = short_window.count('P')
    short_total = short_b_count + short_p_count
    short_b_ratio = short_b_count / short_total if short_total > 0 else 0.5
    short_p_ratio = short_p_count / short_total if short_total > 0 else 0.5
    
    total_b_count = roadmap.count('B')
    total_p_count = roadmap.count('P')
    total_ratio = total_b_count / (total_b_count + total_p_count) if (total_b_count + total_p_count) > 0 else 0.5
    
    trend_window = roadmap[-10:] if len(roadmap) >= 10 else roadmap
    trend_changes = 0
    for j in range(1, len(trend_window)):
        if trend_window[j] != trend_window[j-1]:
            trend_changes += 1
    trend_volatility = trend_changes / len(trend_window) if len(trend_window) > 0 else 0
    
    basic_features = [
        b_ratio, p_ratio, 
        short_b_ratio, short_p_ratio,
        total_ratio,
        trend_volatility,
        streak, streak_type, prev_result
    ]
    
    # 獲取衍生路特徵
    derived_features = analyzer.get_derived_road_features()
    
    # 合併所有特徵
    all_features = np.array(basic_features + derived_features).reshape(1, -1)
    
    # 標準化特徵
    features_scaled = scaler.transform(all_features)
    
    # 進行預測
    pred_prob = model.predict_proba(features_scaled)[0]
    prediction = REVERSE_MAP[np.argmax(pred_prob)]
    probability = float(np.max(pred_prob))
    
    # 如果信心不足，建議觀望
    if probability < 0.55:
        prediction = "觀望"
        
    return prediction, float(pred_prob[0]), float(pred_prob[1]), probability

def detect_dragon(roadmap):
    DRAGON_THRESHOLD = 3
    if len(roadmap) < DRAGON_THRESHOLD: 
        return None, 0
        
    last_result = roadmap[-1]
    streak_len = 0
    for result in reversed(roadmap):
        if result == last_result: 
            streak_len += 1
        else: 
            break
            
    if streak_len >= DRAGON_THRESHOLD: 
        return last_result, streak_len
        
    return None, 0

def calculate_betting_plan(principal, prediction, confidence):
    """計算注碼策略"""
    betting_plan = []
    levels = [
        {"level": 1, "percentage": 0.05, "name": "第一關"},
        {"level": 2, "percentage": 0.10, "name": "第二關"},
        {"level": 3, "percentage": 0.20, "name": "第三關"},
        {"level": 4, "percentage": 0.40, "name": "第四關"}
    ]
    
    for level in levels:
        amount = int(principal * level["percentage"])
        betting_plan.append({
            "level": level["level"],
            "name": level["name"],
            "percentage": level["percentage"],
            "amount": amount,
            "suggestion": prediction if level["level"] == 1 else ("跟注" if level["level"] > 1 else prediction)
        })
    
    return betting_plan

def calculate_profit(user_id):
    """計算用戶盈利"""
    if user_id not in user_sessions:
        return 0
    
    session = user_sessions[user_id]
    if "bets" not in session or not session["bets"]:
        return 0
    
    total_profit = 0
    for bet in session["bets"]:
        if bet["result"] == "win":
            total_profit += bet["amount"]  # 1賠1
        elif bet["result"] == "lose":
            total_profit -= bet["amount"]
    
    return total_profit

def parse_chinese_roadmap(user_input):
    """解析中文牌路輸入"""
    # 移除所有空格和換行符
    cleaned_input = re.sub(r'\s+', '', user_input)
    
    # 分割輸入
    if '，' in cleaned_input:
        roadmap = cleaned_input.split('，')
    elif ',' in cleaned_input:
        roadmap = cleaned_input.split(',')
    else:
        # 處理連續中文輸入
        roadmap = []
        for char in cleaned_input:
            if char in CHINESE_MAP:
                roadmap.append(char)
    
    # 轉換為英文代碼
    english_roadmap = []
    for result in roadmap:
        if result in CHINESE_MAP:
            english_roadmap.append(CHINESE_MAP[result])
        elif result in ['B', 'P', 'T']:
            english_roadmap.append(result)
    
    return english_roadmap

def check_daily_usage(user_id):
    """檢查用戶每日使用時間"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    if user_id not in user_daily_usage:
        user_daily_usage[user_id] = {
            "date": today,
            "usage_seconds": 0,
            "last_start": None
        }
        return True, 0
    
    user_usage = user_daily_usage[user_id]
    
    # 檢查是否是新的一天
    if user_usage["date"] != today:
        user_usage["date"] = today
        user_usage["usage_seconds"] = 0
        user_usage["last_start"] = None
        return True, 0
    
    # 檢查是否超過15分鐘（900秒）
    if user_usage["usage_seconds"] >= 900:
        return False, user_usage["usage_seconds"]
    
    return True, 900 - user_usage["usage_seconds"]

def update_daily_usage(user_id, seconds):
    """更新用戶每日使用時間"""
    if user_id not in user_daily_usage:
        today = datetime.now().strftime("%Y-%m-%d")
        user_daily_usage[user_id] = {
            "date": today,
            "usage_seconds": seconds,
            "last_start": None
        }
    else:
        user_daily_usage[user_id]["usage_seconds"] += seconds

# =============================================================================
# API Endpoint
# =============================================================================
@app.route("/", methods=["GET"])
def home(): 
    return jsonify({"status": "online"})

@app.route('/health', methods=['GET'])
def health_check(): 
    return jsonify({"status": "healthy"})

@app.route("/linebot/predict", methods=["POST"])
def linebot_predict():
    """LINE BOT 專用預測端點"""
    if not models_loaded: 
        load_all_models()
        
    if not models: 
        return jsonify({"error": "模型檔案遺失或損毀。"}), 503
        
    try:
        data = request.get_json()
        user_input = data.get("roadmap", "")
        principal = data.get("principal", 5000)
        user_id = data.get("user_id", "unknown")
        
        # 檢查每日使用時間
        can_use, remaining = check_daily_usage(user_id)
        if not can_use:
            return jsonify({
                "error": "daily_limit_exceeded",
                "message": f"今日使用時間已達15分鐘上限，請明天再使用。"
            }), 429
        
        # 解析中文牌路輸入
        roadmap = parse_chinese_roadmap(user_input)
        
        if not roadmap:
            return jsonify({
                "prediction": "數據不足",
                "message": "請提供有效的牌路數據（庄、闲、和）",
                "confidence": 0.5,
                "banker_prob": 0.5,
                "player_prob": 0.5,
                "tie_prob": 0.0,
                "betting_plan": []
            })
        
        # 初始化用戶會話
        if user_id not in user_sessions:
            user_sessions[user_id] = {
                "start_time": time.time(),
                "principal": principal,
                "bets": [],
                "roadmap": roadmap
            }
        else:
            # 更新用戶會話
            user_sessions[user_id]["principal"] = principal
            user_sessions[user_id]["roadmap"] = roadmap
        
        # 檢查數據是否足夠
        if len(roadmap) < N_FEATURES_WINDOW:
            return jsonify({
                "prediction": "數據不足",
                "message": f"需要至少 {N_FEATURES_WINDOW} 局歷史數據，當前只有 {len(roadmap)} 局",
                "confidence": 0.5,
                "banker_prob": 0.5,
                "player_prob": 0.5,
                "tie_prob": 0.0,
                "betting_plan": []
            })
        
        # 獲取XGBoost預測
        xgb_suggestion, banker_prob, player_prob, xgb_prob = get_ml_prediction(
            models['xgb'], models['scaler'], roadmap
        )
        
        # 檢查長龍
        dragon_type, streak_len = detect_dragon(roadmap)
        if dragon_type:
            dragon_vote = 'B' if dragon_type == 'B' else 'P'
            BREAK_DRAGON_CONFIDENCE = 0.70
            
            if xgb_suggestion != dragon_vote and xgb_prob > BREAK_DRAGON_CONFIDENCE:
                pass  # 保持原預測
            else:
                xgb_suggestion = dragon_vote
                xgb_prob = max(xgb_prob, 0.6)
        
        # 計算和局概率
        tie_prob = 0.05
        
        # 正規化莊閒概率
        total = banker_prob + player_prob
        if total > 0:
            banker_prob = banker_prob / total * (1 - tie_prob)
            player_prob = player_prob / total * (1 - tie_prob)
        
        # 計算注碼策略
        betting_plan = calculate_betting_plan(principal, xgb_suggestion, xgb_prob)
        
        # 更新使用時間（假設每次預測使用10秒）
        update_daily_usage(user_id, 10)
        
        # 為LINE BOT簡化響應格式
        return jsonify({
            "prediction": xgb_suggestion,
            "confidence": round(xgb_prob, 2),
            "banker_prob": round(banker_prob, 4),
            "player_prob": round(player_prob, 4),
            "tie_prob": round(tie_prob, 4),
            "dragon": dragon_type if dragon_type else None,
            "streak": streak_len if dragon_type else 0,
            "betting_plan": betting_plan,
            "roadmap_length": len(roadmap),
            "daily_remaining": remaining - 10
        })
    except Exception as e:
        app.logger.error(f"LINE BOT 預測時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": "內部伺服器錯誤"}), 500

@app.route("/linebot/check_usage", methods=["POST"])
def linebot_check_usage():
    """檢查用戶使用時間"""
    try:
        data = request.get_json()
        user_id = data.get("user_id", "unknown")
        
        can_use, remaining = check_daily_usage(user_id)
        
        if not can_use:
            return jsonify({
                "can_use": False,
                "message": f"今日使用時間已達15分鐘上限，請明天再使用。"
            })
        
        return jsonify({
            "can_use": True,
            "remaining": remaining,
            "message": f"今日還剩 {int(remaining // 60)}分{int(remaining % 60)}秒可用時間"
        })
            
    except Exception as e:
        app.logger.error(f"檢查使用時間時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": "內部伺服器錯誤"}), 500

@app.route("/linebot/record_bet", methods=["POST"])
def linebot_record_bet():
    """記錄下注結果"""
    try:
        data = request.get_json()
        user_id = data.get("user_id", "unknown")
        amount = data.get("amount", 0)
        prediction = data.get("prediction", "")
        actual_result = data.get("actual_result", "")
        
        if user_id not in user_sessions:
            return jsonify({"error": "會話不存在"}), 404
        
        # 確定輸贏
        result = "win" if prediction == actual_result else "lose"
        
        # 記錄下注
        if "bets" not in user_sessions[user_id]:
            user_sessions[user_id]["bets"] = []
        
        user_sessions[user_id]["bets"].append({
            "time": time.time(),
            "amount": amount,
            "prediction": prediction,
            "actual_result": actual_result,
            "result": result
        })
        
        # 更新使用時間（假設記錄結果使用5秒）
        update_daily_usage(user_id, 5)
        
        return jsonify({"message": "下注記錄成功"})
        
    except Exception as e:
        app.logger.error(f"記錄下注時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": "內部伺服器錯誤"}), 500

@app.route("/linebot/learn", methods=["POST"])
def linebot_learn():
    """學習新的牌路數據"""
    try:
        data = request.get_json()
        user_input = data.get("roadmap", "")
        
        # 解析中文牌路輸入
        roadmap = parse_chinese_roadmap(user_input)
        
        if not roadmap:
            return jsonify({"error": "無效的牌路數據"}), 400
        
        # 加載現有數據
        existing_data = load_history_data()
        
        # 添加新數據
        existing_data.extend(roadmap)
        
        # 保存數據
        save_history_data(roadmap)
        
        return jsonify({
            "message": f"成功學習 {len(roadmap)} 筆新數據，總數據量: {len(existing_data)}",
            "total_data": len(existing_data)
        })
        
    except Exception as e:
        app.logger.error(f"學習數據時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": "內部伺服器錯誤"}), 500

# 導入並註冊 LINE BOT 藍圖
from linebot import linebot_bp
app.register_blueprint(linebot_bp)

if __name__ == "__main__":
    load_all_models()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    load_all_models()
