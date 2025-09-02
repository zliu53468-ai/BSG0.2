# -*- coding: utf-8 -*-
import os
import json
import time
import random
from flask import Blueprint, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction

# 初始化 LINE BOT
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN', ''))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET', ''))

# 創建藍圖
linebot_bp = Blueprint('linebot', __name__)

# 用戶狀態跟蹤
user_states = {}

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
    user_id = event.source.user_id
    user_message = event.message.text
    
    # 檢查會話狀態
    session_check = check_user_session(user_id)
    if not session_check.get("active", False):
        if "message" in session_check:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=session_check["message"])
            )
        return
    
    # 初始化用戶狀態
    if user_id not in user_states:
        user_states[user_id] = {
            "step": "waiting_roadmap",
            "principal": 5000,
            "roadmap": []
        }
    
    # 處理本金選擇
    if user_message in ['5000', '10000', '15000', '20000', '30000', '50000']:
        user_states[user_id]["principal"] = int(user_message)
        user_states[user_id]["step"] = "waiting_roadmap"
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=f"✅ 已設定本金為 {user_message} 元\n\n請輸入路單數據，格式為: B,P,B,P,B,P,...",
                quick_reply=QuickReply(items=[])
            )
        )
        return
    
    # 處理幫助指令
    if user_message.lower() in ['help', '幫助', 'menu', '選單']:
        show_main_menu(event, user_id)
        return
    
    # 處理路單數據
    if ',' in user_message:
        # 處理路單數據
        roadmap = [x.strip().upper() for x in user_message.split(',')]
        user_states[user_id]["roadmap"] = roadmap
        user_states[user_id]["step"] = "waiting_principal"
        
        # 顯示本金選擇菜單
        show_principal_menu(event)
        return
    
    # 處理預測指令
    if user_message.lower() in ['預測', 'predict', '分析', 'analyze']:
        if not user_states[user_id]["roadmap"]:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="請先提供路單數據，格式為: B,P,B,P,B,P,...")
            )
            return
        
        # 呼叫預測API
        try:
            import requests
            response = requests.post(
                f"{request.host_url}linebot/predict",
                json={
                    "roadmap": user_states[user_id]["roadmap"],
                    "principal": user_states[user_id]["principal"],
                    "user_id": user_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', '未知')
                confidence = data.get('confidence', 0) * 100
                
                # 構建高科技風格的回覆
                tech_emojis = "🤖🚀💡⚡🎯🔮"
                reply_text = f"{tech_emojis} AI 預測分析中 .... {tech_emojis}\n\n"
                
                if prediction == "觀望":
                    reply_text += f"📊 當前建議: 觀望 (信心度: {confidence:.1f}%)\n\n"
                    reply_text += "建議暫時不要下注，等待更明確的信號。"
                else:
                    # 添加下注建議
                    chinese_prediction = "莊" if prediction == "B" else "閒"
                    reply_text += f"🎯 下注建議: {chinese_prediction} (信心度: {confidence:.1f}%)\n\n"
                    
                    # 添加長龍信息
                    dragon_info = ""
                    if data.get('dragon'):
                        dragon_type = "莊" if data['dragon'] == "B" else "閒"
                        dragon_info = f"🐉 檢測到長龍: {dragon_type} (連續 {data['streak']} 次)\n\n"
                    
                    reply_text += dragon_info
                    
                    # 添加注碼策略
                    reply_text += "💰 本金配注策略:\n"
                    betting_plan = data.get('betting_plan', [])
                    for plan in betting_plan:
                        chinese_suggestion = "莊" if plan['suggestion'] == "B" else "閒" if plan['suggestion'] == "P" else plan['suggestion']
                        reply_text += f"{plan['name']}: {plan['amount']}元 ({plan['percentage']*100:.0f}%) → {chinese_suggestion}\n"
                    
                    reply_text += "\n⚡ 策略說明: 採用過關式注碼，如第一關未過則進入下一關"
                
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(
                        text=reply_text,
                        quick_reply=QuickReply(items=[
                            QuickReplyButton(
                                action=MessageAction(label="更改本金", text="menu")
                            ),
                            QuickReplyButton(
                                action=MessageAction(label="重新預測", text="predict")
                            ),
                            QuickReplyButton(
                                action=MessageAction(label="記錄結果", text="記錄結果")
                            )
                        ])
                    )
                )
            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="❌ 預測服務暫時不可用，請稍後再試")
                )
                
        except Exception as e:
            print(f"預測錯誤: {e}")
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ 預測時發生錯誤，請稍後再試")
            )
    
    # 處理記錄結果
    elif user_message == "記錄結果":
        show_result_menu(event, user_id)
    
    # 處理結果記錄
    elif user_message in ["莊贏", "閒贏", "和局"]:
        record_bet_result(event, user_id, user_message)
    
    else:
        # 顯示主選單
        show_main_menu(event, user_id)

def check_user_session(user_id):
    """檢查用戶會話狀態"""
    try:
        import requests
        response = requests.post(
            f"{request.host_url}linebot/check_session",
            json={"user_id": user_id},
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"active": True}  # 默認允許訪問
    except:
        return {"active": True}  # 默認允許訪問

def record_bet_result(event, user_id, result):
    """記錄下注結果"""
    try:
        import requests
        
        # 獲取最後一次預測
        prediction_response = requests.post(
            f"{request.host_url}linebot/predict",
            json={
                "roadmap": user_states[user_id]["roadmap"],
                "principal": user_states[user_id]["principal"],
                "user_id": user_id
            },
            timeout=10
        )
        
        if prediction_response.status_code != 200:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ 無法獲取預測信息")
            )
            return
        
        prediction_data = prediction_response.json()
        prediction = prediction_data.get('prediction', '')
        
        if prediction == "觀望":
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ 當前預測為觀望，無需記錄結果")
            )
            return
        
        # 確定下注金額（使用第一關金額）
        betting_plan = prediction_data.get('betting_plan', [])
        if not betting_plan:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ 無法獲取下注計劃")
            )
            return
        
        amount = betting_plan[0]["amount"]  # 第一關金額
        
        # 確定實際結果
        actual_result = ""
        if result == "莊贏":
            actual_result = "B"
        elif result == "閒贏":
            actual_result = "P"
        else:
            actual_result = "T"
        
        # 記錄下注
        record_response = requests.post(
            f"{request.host_url}linebot/record_bet",
            json={
                "user_id": user_id,
                "amount": amount,
                "prediction": prediction,
                "actual_result": actual_result
            },
            timeout=5
        )
        
        if record_response.status_code == 200:
            # 確定輸贏
            win = (prediction == actual_result)
            
            # 構建回覆
            tech_emojis = "📊🔢✅❌"
            reply_text = f"{tech_emojis} 下注結果記錄完成 {tech_emojis}\n\n"
            reply_text += f"預測: {'莊' if prediction == 'B' else '閒'}\n"
            reply_text += f"結果: {result}\n"
            reply_text += f"金額: {amount}元\n"
            reply_text += f"狀態: {'✅ 贏' if win else '❌ 輸'}\n\n"
            
            # 檢查會話狀態
            session_check = check_user_session(user_id)
            if not session_check.get("active", False):
                reply_text += f"\n{session_check.get('message', '')}"
            else:
                remaining_time = session_check.get('remaining_time', 0)
                reply_text += f"⏰ 剩餘時間: {int(remaining_time // 60)}分{int(remaining_time % 60)}秒"
            
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=reply_text,
                    quick_reply=QuickReply(items=[
                        QuickReplyButton(
                            action=MessageAction(label="繼續預測", text="predict")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="查看狀態", text="status")
                        )
                    ])
                )
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ 記錄結果時發生錯誤")
            )
            
    except Exception as e:
        print(f"記錄結果錯誤: {e}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="❌ 記錄結果時發生錯誤")
        )

def show_main_menu(event, user_id):
    """顯示主選單"""
    # 檢查會話狀態
    session_check = check_user_session(user_id)
    remaining_time = session_check.get('remaining_time', 0)
    
    menu_text = "🎯 歡迎使用百家樂AI預測系統\n\n"
    menu_text += f"⏰ 剩餘時間: {int(remaining_time // 60)}分{int(remaining_time % 60)}秒\n\n"
    menu_text += "請選擇操作:"
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=menu_text,
            quick_reply=QuickReply(items=[
                QuickReplyButton(
                    action=MessageAction(label="設定本金", text="menu")
                ),
                QuickReplyButton(
                    action=MessageAction(label="開始預測", text="predict")
                ),
                QuickReplyButton(
                    action=MessageAction(label="使用說明", text="help")
                )
            ])
        )
    )

def show_principal_menu(event):
    """顯示本金選擇菜單"""
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text="請選擇本金金額:",
            quick_reply=QuickReply(items=[
                QuickReplyButton(
                    action=MessageAction(label="5000", text="5000")
                ),
                QuickReplyButton(
                    action=MessageAction(label="10000", text="10000")
                ),
                QuickReplyButton(
                    action=MessageAction(label="15000", text="15000")
                ),
                QuickReplyButton(
                    action=MessageAction(label="20000", text="20000")
                ),
                QuickReplyButton(
                    action=MessageAction(label="30000", text="30000")
                ),
                QuickReplyButton(
                    action=MessageAction(label="50000", text="50000")
                )
            ])
        )
    )

def show_result_menu(event, user_id):
    """顯示結果記錄菜單"""
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text="請選擇本局結果:",
            quick_reply=QuickReply(items=[
                QuickReplyButton(
                    action=MessageAction(label="莊贏", text="莊贏")
                ),
                QuickReplyButton(
                    action=MessageAction(label="閒贏", text="閒贏")
                ),
                QuickReplyButton(
                    action=MessageAction(label="和局", text="和局")
                )
            ])
        )
    )
