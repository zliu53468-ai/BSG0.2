**檔案名稱**：`line_bot_webhook.py`
```python
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from linebot import LineBotApi, WebhookHandler
from linebot.models import *
from linebot.exceptions import InvalidSignatureError
import redis
import joblib

# ==================== 環境驗證 ====================
REQUIRED_ENV = ['LINE_CHANNEL_ACCESS_TOKEN', 'LINE_CHANNEL_SECRET', 'REDIS_URL']
if missing := [var for var in REQUIRED_ENV if not os.getenv(var)]:
    raise EnvironmentError(f"缺少環境變數: {', '.join(missing)}")

# ==================== 全局初始化 ====================
r = redis.from_url(os.getenv('REDIS_URL'))
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
linebot_bp = Blueprint('linebot', __name__, url_prefix='/linebot')

# ==================== 模型載入 ====================
MODEL_MAP = {
    'B': '庄', 
    'P': '闲', 
    'T': '和'
}

try:
    xgboost_model = joblib.load('models/xgboost_model.pkl')
    lgb_model = joblib.load('models/lgb_model.pkl')
    hmm_model = joblib.load('models/hmm_model.pkl')
except Exception as e:
    raise RuntimeError(f"模型載入失敗: {str(e)}")

# ==================== Webhook 強化版 ====================
@linebot_bp.route("/webhook", methods=['POST'])
def webhook():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    
    # IP速率限制 (60次/分鐘)
    client_ip = request.remote_addr
    rate_key = f"rate:{client_ip}"
    if r.incr(rate_key) > 60:
        return jsonify({"error": "請求過於頻繁"}), 429
    r.expire(rate_key, 60)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return jsonify({"error": "簽章無效"}), 403
    except Exception as e:
        current_app.logger.error(f"Webhook錯誤: {str(e)}")
        return jsonify({"error": "伺服器錯誤"}), 500
    
    return jsonify({"status": "success"})

# ==================== 狀態管理 (Redis版) ====================
def get_user_state(user_id):
    if state := r.get(f"user:{user_id}"):
        return json.loads(state)
    return {"step": "init", "principal": 5000, "roadmap": ""}

def update_user_state(user_id, state):
    r.setex(f"user:{user_id}", 1800, json.dumps(state))

# ==================== 核心預測邏輯 ====================
def analyze_roadmap(roadmap):
    # 特徵工程 (與訓練時完全一致)
    features = np.array([[1 if c == 'B' else 0 for c in roadmap[-20:]] + [roadmap.count('B')]])
    
    # 集成模型預測
    xgb_proba = xgboost_model.predict_proba(features)[0]
    lgb_proba = lgb_model.predict_proba(features)[0]
    final_proba = (xgb_proba + lgb_proba) / 2
    
    # HMM狀態分析
    hmm_state = hmm_model.predict(features.reshape(-1, 1))
    
    return {
        "prediction": MODEL_MAP[np.argmax(final_proba)],
        "confidence": float(np.max(final_proba)),
        "trend": "上升" if hmm_state[0] == 1 else "下降"
    }

# ==================== 訊息處理器 ====================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    msg = event.message.text
    state = get_user_state(user_id)
    
    # 狀態機邏輯 (精簡版)
    if msg == "開始":
        state["step"] = "waiting_roadmap"
        reply = TextSendMessage(text="請輸入牌路走勢")
    elif state["step"] == "waiting_roadmap":
        state.update({"roadmap": msg, "step": "waiting_principal"})
        reply = TextSendMessage(
            text="選擇本金",
            quick_reply=QuickReply(items=[
                QuickReplyButton(action=MessageAction(label=str(amt), text=str(amt)))
                for amt in [5000, 10000, 15000, 20000, 30000, 50000]
            ])
        )
    elif msg.isdigit() and int(msg) in [5000, 10000, 15000, 20000, 30000, 50000]:
        state.update({"principal": int(msg), "step": "ready"})
        result = analyze_roadmap(state["roadmap"])
        reply = TextSendMessage(
            text=f"BGS建議: {result['prediction']} (信心度: {result['confidence']:.0%})\n趨勢: {result['trend']}"
        )
    else:
        reply = TextSendMessage(text="請輸入有效指令")
    
    update_user_state(user_id, state)
    line_bot_api.reply_message(event.reply_token, reply)

# 其他輔助函式保持不變...
```
