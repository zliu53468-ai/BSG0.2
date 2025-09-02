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
    
    # 檢查每日使用時間
    usage_check = check_user_usage(user_id)
    if not usage_check.get("can_use", False):
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=usage_check.get("message", "今日使用時間已達上限"))
        )
        return
    
    # 初始化用戶狀態
    if user_id not in user_states:
        user_states[user_id] = {
            "step": "waiting_roadmap",
            "principal": 5000,
            "roadmap": ""
        }
        send_welcome_message(event)
        return
    
    # 處理幫助指令
    if user_message.lower() in ['help', '幫助', 'menu', '選單', '說明']:
        show_help_menu(event, user_id)
        return
    
    # 處理開始指令
    if user_message in ['開始', 'start', 'go', '分析']:
        start_prediction(event, user_id)
        return
    
    # 處理本金選擇
    if user_message in ['5000', '10000', '15000', '20000', '30000', '50000']:
        user_states[user_id]["principal"] = int(user_message)
        
        if user_states[user_id]["step"] == "waiting_principal":
            user_states[user_id]["step"] = "ready"
            # 進行預測
            make_prediction(event, user_id)
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"✅ 已設定本金為 {user_message} 元\n\n請輸入牌路走勢（例如: 庄,闲,庄,庄,闲,和）")
            )
        return
    
    # 處理學習指令
    if user_message.lower() in ['learn', '學習']:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請輸入要學習的牌路數據（例如: 庄,闲,庄,庄,闲,和）")
        )
        user_states[user_id]["step"] = "waiting_learn"
        return
    
    # 處理狀態查詢
    if user_message.lower() in ['status', '狀態']:
        show_status(event, user_id)
        return
    
    # 處理範例指令
    if user_message in ['範例', 'example', '例子']:
        show_example(event, user_id)
        return
    
    # 處理學習數據
    if user_states[user_id]["step"] == "waiting_learn":
        learn_roadmap(event, user_id, user_message)
        return
    
    # 處理牌路輸入
    if user_states[user_id]["step"] == "waiting_roadmap":
        user_states[user_id]["roadmap"] = user_message
        user_states[user_id]["step"] = "waiting_principal"
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="✅ 已記錄牌路數據\n\n請選擇本金金額:",
                quick_reply=QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="5000", text="5000")),
                    QuickReplyButton(action=MessageAction(label="10000", text="10000")),
                    QuickReplyButton(action=MessageAction(label="15000", text="15000")),
                    QuickReplyButton(action=MessageAction(label="20000", text="20000")),
                    QuickReplyButton(action=MessageAction(label="30000", text="30000")),
                    QuickReplyButton(action=MessageAction(label="50000", text="50000"))
                ])
            )
        )
        return
    
    # 處理重新預測
    if user_message.lower() in ['predict', '預測', '重新預測']:
        user_states[user_id]["step"] = "waiting_roadmap"
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請輸入當前牌路走勢（例如: 庄,闲,庄,庄,闲,和）")
        )
        return
    
    # 處理記錄結果
    if user_message in ["庄贏", "閒贏", "和局"]:
        record_bet_result(event, user_id, user_message)
        return
    
    # 默認回應 - 要求輸入牌路
    user_states[user_id]["step"] = "waiting_roadmap"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="🎯 請輸入當前牌路走勢（例如: 庄,闲,庄,庄,闲,和）")
    )

def send_welcome_message(event):
    """發送歡迎消息"""
    welcome_text = """
🤖🚀✨ 歡迎使用【BGS AI分析系統】 ✨🚀🤖

🎯 我是您的專業分析助手，提供數據驅動的預測與策略建議！

⭐️ 【使用說明】：
1️⃣ 輸入當前牌路走勢（格式：庄,闲,庄,和 或 庄闲庄和）
2️⃣ 選擇本金金額（5000-50000）
3️⃣ 獲得分析結果與注碼策略
4️⃣ 記錄實際結果幫助系統優化

💡 您可以隨時輸入「幫助」查看詳細說明

📊 請輸入當前牌路走勢開始使用：
（例如：庄,闲,庄,庄,闲,和）
    """
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=welcome_text,
            quick_reply=QuickReply(items=[
                QuickReplyButton(action=MessageAction(label="查看幫助", text="幫助")),
                QuickReplyButton(action=MessageAction(label="開始分析", text="開始")),
                QuickReplyButton(action=MessageAction(label="使用範例", text="範例"))
            ])
        )
    )

def start_prediction(event, user_id):
    """開始預測流程"""
    user_states[user_id]["step"] = "waiting_roadmap"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text="🎯 請輸入當前牌路走勢：\n（例如：庄,闲,庄,庄,闲,和 或 庄闲庄庄闲和）",
            quick_reply=QuickReply(items=[
                QuickReplyButton(action=MessageAction(label="使用範例", text="範例")),
                QuickReplyButton(action=MessageAction(label="查看幫助", text="幫助"))
            ])
        )
    )

def make_prediction(event, user_id):
    """進行預測"""
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
            
            # 構建BGS AI風格的回覆
            bgs_emojis = "🔮📊🎯⚡"
            reply_text = f"{bgs_emojis} BGS AI分析完成 {bgs_emojis}\n\n"
            
            if prediction == "觀望":
                reply_text += f"📊 當前建議: 觀望 (信心度: {confidence:.1f}%)\n\n"
                reply_text += "建議暫時不要下注，等待更明確的信號。"
            else:
                # 添加下注建議
                chinese_prediction = "庄" if prediction == "B" else "闲"
                reply_text += f"🎯 BGS建議: {chinese_prediction} (信心度: {confidence:.1f}%)\n\n"
                
                # 添加長龍信息
                dragon_info = ""
                if data.get('dragon'):
                    dragon_type = "庄" if data['dragon'] == "B" else "闲"
                    dragon_info = f"📈 檢測到趨勢: {dragon_type} (連續 {data['streak']} 次)\n\n"
                
                reply_text += dragon_info
                
                # 添加注碼策略
                reply_text += "💰 BGS資金分配建議:\n"
                betting_plan = data.get('betting_plan', [])
                for plan in betting_plan:
                    chinese_suggestion = "庄" if plan['suggestion'] == "B" else "闲" if plan['suggestion'] == "P" else plan['suggestion']
                    reply_text += f"{plan['name']}: {plan['amount']}元 ({plan['percentage']*100:.0f}%) → {chinese_suggestion}\n"
                
                reply_text += "\n⚡ BGS策略: 採用漸進式注碼，如第一關未過則進入下一關"
                
                # 添加每日剩餘時間
                remaining = data.get('daily_remaining', 0)
                reply_text += f"\n⏰ 今日剩餘時間: {int(remaining // 60)}分{int(remaining % 60)}秒"
            
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=reply_text,
                    quick_reply=QuickReply(items=[
                        QuickReplyButton(action=MessageAction(label="記錄結果", text="記錄結果")),
                        QuickReplyButton(action=MessageAction(label="重新分析", text="開始")),
                        QuickReplyButton(action=MessageAction(label="查看狀態", text="狀態"))
                    ])
                )
            )
        elif response.status_code == 429:
            # 每日使用時間已達上限
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ 今日使用時間已達15分鐘上限，請明天再使用。")
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ BGS分析服務暫時不可用，請稍後再試")
            )
            
    except Exception as e:
        print(f"分析錯誤: {e}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="❌ BGS分析時發生錯誤，請稍後再試")
        )

def learn_roadmap(event, user_id, user_message):
    """學習新的牌路數據"""
    try:
        import requests
        
        response = requests.post(
            f"{request.host_url}linebot/learn",
            json={"roadmap": user_message},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"✅ {data['message']}")
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ 學習數據時發生錯誤")
            )
        
        user_states[user_id]["step"] = "waiting_roadmap"
        
    except Exception as e:
        print(f"學習數據錯誤: {e}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="❌ 學習數據時發生錯誤")
        )
        user_states[user_id]["step"] = "waiting_roadmap"

def show_help_menu(event, user_id):
    """顯示幫助選單"""
    help_text = """
📖 【BGS AI分析系統 - 幫助指南】🤖

⭐️ 主要功能：
🎯 數據分析 - 基于先進算法分析牌路趨勢
💰 注碼策略 - 提供科學的資金管理建議
📊 學習優化 - 不斷提升預測準確性
⏰ 時間管理 - 每日15分鐘使用限制

⭐️ 使用流程：
1. 輸入牌路 → 2. 選擇本金 → 3. 獲得分析 → 4. 記錄結果

⭐️ 常用指令：
• 「開始」 - 開始分析
• 「幫助」 - 顯示此幫助
• 「狀態」 - 查看當前狀態
• 「學習」 - 提交數據幫助系統學習
• 「範例」 - 查看使用範例

⭐️ 牌路格式：
庄,闲,庄,庄,闲,和 或 庄闲庄庄闲和

⏰ 每日限制：15分鐘使用時間
    """
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=help_text,
            quick_reply=QuickReply(items=[
                QuickReplyButton(action=MessageAction(label="開始分析", text="開始")),
                QuickReplyButton(action=MessageAction(label="查看範例", text="範例")),
                QuickReplyButton(action=MessageAction(label="查看狀態", text="狀態"))
            ])
        )
    )

def show_example(event, user_id):
    """顯示使用範例"""
    example_text = """
📝 【BGS AI分析系統 - 使用範例】：

1. 輸入牌路：
   庄,闲,庄,庄,闲,和
   或
   庄闲庄庄闲和

2. 選擇本金：
   5000/10000/15000/20000/30000/50000

3. 獲得分析：
   📊 BGS系統會分析牌路並給出建議

4. 記錄結果：
   根據實際結果輸入「庄贏」、「闲贏」或「和局」

💡 提示：您輸入的數據越多，BGS分析越準確！
    """
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=example_text,
            quick_reply=QuickReply(items=[
                QuickReplyButton(action=MessageAction(label="開始使用", text="開始")),
                QuickReplyButton(action=MessageAction(label="查看幫助", text="幫助"))
            ])
        )
    )

def show_status(event, user_id):
    """顯示當前狀態"""
    try:
        import requests
        
        # 檢查使用時間
        usage_check = check_user_usage(user_id)
        
        status_text = "📊 BGS AI分析系統 - 當前狀態\n\n"
        
        if usage_check.get("can_use", False):
            remaining = usage_check.get("remaining", 0)
            status_text += f"⏰ 今日剩餘時間: {int(remaining // 60)}分{int(remaining % 60)}秒\n"
            
            if user_id in user_states:
                status_text += f"💰 設定本金: {user_states[user_id]['principal']}元\n"
            
            status_text += "\n🔄 輸入「開始」進行BGS分析"
        else:
            status_text = usage_check.get("message", "今日使用時間已達上限")
                
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=status_text)
        )
            
    except Exception as e:
        print(f"獲取狀態錯誤: {e}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="❌ 獲取BGS狀態時發生錯誤")
        )

def check_user_usage(user_id):
    """檢查用戶使用時間"""
    try:
        import requests
        response = requests.post(
            f"{request.host_url}linebot/check_usage",
            json={"user_id": user_id},
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"can_use": True}
    except:
        return {"can_use": True}

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
        if result == "庄贏":
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
            bgs_emojis = "📊🔢✅❌"
            reply_text = f"{bgs_emojis} 下注結果記錄完成 {bgs_emojis}\n\n"
            reply_text += f"預測: {'庄' if prediction == 'B' else '闲'}\n"
            reply_text += f"結果: {result}\n"
            reply_text += f"金額: {amount}元\n"
            reply_text += f"狀態: {'✅ 贏' if win else '❌ 輸'}\n\n"
            
            # 檢查使用時間
            usage_check = check_user_usage(user_id)
            if usage_check.get("can_use", False):
                remaining = usage_check.get("remaining", 0)
                reply_text += f"⏰ 今日剩餘時間: {int(remaining // 60)}分{int(remaining % 60)}秒"
            else:
                reply_text += f"⏰ {usage_check.get('message', '今日使用時間已達上限')}"
            
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=reply_text,
                    quick_reply=QuickReply(items=[
                        QuickReplyButton(action=MessageAction(label="繼續預測", text="開始")),
                        QuickReplyButton(action=MessageAction(label="查看狀態", text="狀態"))
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
