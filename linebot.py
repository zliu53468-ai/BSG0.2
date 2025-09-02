# -*- coding: utf-8 -*-
import os
import json
from flask import Blueprint, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# 初始化 LINE BOT
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN', ''))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET', ''))

# 創建藍圖
linebot_bp = Blueprint('linebot', __name__)

@linebot_bp.route("/webhook", methods=['POST'])
def webhook():
    # 獲取請求標頭中的簽名驗證
    signature = request.headers['X-Line-Signature']
    
    # 獲取請求體內容
    body = request.get_data(as_text=True)
    
    try:
        # 處理 webhook 體
        handler.handle(body, signature)
    except Exception as e:
        print(f"Error: {e}")
    
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # 獲取用戶發送的訊息
    user_message = event.message.text
    
    # 簡單的指令處理
    if user_message.lower() in ['help', '幫助']:
        reply_text = """歡迎使用百家樂預測機器人！
        
指令列表：
- 「預測」或「predict」: 進行下一局預測
- 「分析」或「analyze」: 獲取詳細分析
- 「幫助」或「help」: 顯示此幫助訊息
        
請先輸入路單數據，格式為: B,P,B,P,B,P,..."""
    
    elif user_message.lower() in ['預測', 'predict']:
        reply_text = "請先提供路單數據，格式為: B,P,B,P,B,P,..."
    
    elif user_message.lower() in ['分析', 'analyze']:
        reply_text = "請先提供路單數據，格式為: B,P,B,P,B,P,..."
    
    elif ',' in user_message:
        # 處理路單數據
        roadmap = [x.strip().upper() for x in user_message.split(',')]
        
        # 呼叫預測API
        try:
            import requests
            response = requests.post(
                f"{request.host_url}linebot/predict",
                json={"roadmap": roadmap},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', '未知')
                confidence = data.get('confidence', 0) * 100
                
                if prediction == "觀望":
                    reply_text = f"建議觀望 (信心度: {confidence:.1f}%)"
                else:
                    dragon_info = ""
                    if data.get('dragon'):
                        dragon_info = f"\n檢測到長龍: {data['dragon']} (連續 {data['streak']} 次)"
                    
                    reply_text = f"預測下一局: {prediction}\n信心度: {confidence:.1f}%{dragon_info}"
            else:
                reply_text = "預測服務暫時不可用，請稍後再試"
                
        except Exception as e:
            print(f"預測錯誤: {e}")
            reply_text = "預測時發生錯誤，請稍後再試"
    
    else:
        reply_text = "無法識別的指令，請輸入「幫助」查看使用說明"
    
    # 回覆訊息給用戶
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

# 在 app.py 中註冊此藍圖
# from linebot import linebot_bp
# app.register_blueprint(linebot_bp, url_prefix='/linebot')
