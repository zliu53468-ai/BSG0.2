# -*- coding: utf-8 -*-
import numpy as np
import joblib
import os
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from hmmlearn import hmm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# =============================================================================
# 全域設定
# =============================================================================
MODEL_DIR = 'models'
TOTAL_DATA_SIZE = 10000 # 增加數據量以更好地訓練神經網路
N_FEATURES_WINDOW = 20
LSTM_SEQUENCE_LENGTH = 15 # LSTM 模型回看的序列長度
LABEL_MAP = {'B': 0, 'P': 1}

REAL_HISTORY_DATA = [
    "P", "P", "T", "B", "T", "B", "P", "B", "P", "P", "B", "B", "T", "B", "B", "P", "B", "B", "P", "B", "B", "T", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "P", "B", "P", "T", "T", "B", "P", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "B", "B", "B", "P", "P", "B", "P", "B", "B", "P", "P", "P", "P", "P", "P", "B", "B", "T", "B", "T", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "T", "T", "B", "B", "B", "B", "P", "B", "B", "B", "P", "B", "T", "P", "P", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "P", "P", "B", "T", "B", "P", "B", "T", "B", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "B", "T", "T", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "T", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "P", "T", "B", "P", "T", "B", "B", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "B", "B", "P", "B", "P", "B", "T", "B", "B", "P", "P", "B", "B", "P", "T", "P", "P", "B", "P", "B", "B", "B", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "T", "T", "B", "P", "P", "B", "P", "P", "B", "B", "P", "B", "P", "T", "P", "P", "P", "P", "B", "B", "B", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "P", "B", "P", "B", "P", "T", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "B", "T", "B", "T", "B", "P", "B", "P", "T", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "P", "P", "B", "P", "P", "P", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "B", "P", "P", "T", "P", "B", "B", "T", "B", "P", "T", "P", "B", "P", "B", "B", "P", "B", "B", "T", "P", "P", "P", "P", "T", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "T", "P", "P", "T", "P", "P", "B", "P", "P", "B", "P", "P", "B", "P", "P", "T", "B", "P", "B", "P", "P", "B", "B", "B", "B", "T", "T", "T", "B", "B", "B", "B", "B", "B", "P", "P", "P", "T", "P", "T", "B", "P", "P", "T", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "P", "P", "T", "P", "B", "P", "P", "B", "T", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "T", "T", "B", "P", "B", "B", "B", "T", "T", "B", "B", "P", "B", "T", "P", "B", "P", "B", "P", "P", "P", "B", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", "B", "B", "P", "B", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "B", "B", "P", "B", "T", "T", "P", "B", "B", "B", "P", "B", "B", "P", "B", "P", "B", "P", "P", "P", "P", "P", "P", "B", "B", "B", "P", "T", "P", "B", "T", "B", "B", "B", "B", "T", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "B", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "B", "B", "B", "T", "B", "B", "P", "B", "P", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "P", "B", "P", "P", "B", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "B", "B", "B", "B", "B", "B", "P", "T", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "P", "P", "T", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "P", "B", "T", "P", "B", "B", "P", "B", "P", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "T", "P", "P", "B", "T", "B", "P", "P", "P", "B", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "B", "P", "P", "T", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "P", "T", "P", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "T", "T", "P", "T", "B", "T", "P", "T", "P", "T", "P", "P", "B", "B", "P", "P", "P", "P", "P"
]

# =============================================================================
# 路單分析核心 (BaccaratAnalyzer)
# =============================================================================
class BaccaratAnalyzer:
    def __init__(self, roadmap):
        self.roadmap = [r for r in roadmap if r in ['B', 'P']]
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
# 特徵工程與訓練
# =============================================================================
def extract_xgb_features(full_roadmap):
    features_list, labels = [], []
    if len(full_roadmap) <= N_FEATURES_WINDOW: return np.array([]), np.array([])
    for i in range(N_FEATURES_WINDOW, len(full_roadmap)):
        current_roadmap = full_roadmap[:i]
        label = full_roadmap[i]
        if label not in LABEL_MAP: continue
        window = current_roadmap[-N_FEATURES_WINDOW:]
        b_count = window.count('B'); p_count = window.count('P'); total = b_count + p_count
        b_ratio = b_count / total if total > 0 else 0.5; p_ratio = p_count / total if total > 0 else 0.5
        streak = 0; last_result = None
        for item in reversed(window):
            if item in ['B', 'P']:
                if last_result is None: last_result = item; streak = 1
                elif item == last_result: streak += 1
                else: break
        streak_type = LABEL_MAP.get(last_result, -1)
        prev_result = LABEL_MAP.get(window[-1], -1) if window else -1
        basic_features = [b_ratio, p_ratio, streak, streak_type, prev_result]
        analyzer = BaccaratAnalyzer(current_roadmap)
        derived_features = analyzer.get_derived_road_features()
        all_features = basic_features + derived_features
        features_list.append(all_features)
        labels.append(LABEL_MAP[label])
    return np.array(features_list), np.array(labels)

def prepare_lstm_data(roadmap_numeric):
    X, y = [], []
    for i in range(len(roadmap_numeric) - LSTM_SEQUENCE_LENGTH):
        X.append(roadmap_numeric[i:(i + LSTM_SEQUENCE_LENGTH)])
        y.append(roadmap_numeric[i + LSTM_SEQUENCE_LENGTH])
    return np.array(X), np.array(y)

def train():
    print("="*50); print("開始重新訓練 AI 模型 (三體決策引擎)..."); print("="*50)
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR); print(f"✅ 已建立目錄: {MODEL_DIR}")

    roadmap = list(REAL_HISTORY_DATA)
    print(f"🔄 已載入 {len(roadmap)} 筆真實歷史數據。")
    num_synthetic_needed = TOTAL_DATA_SIZE - len(roadmap)
    if num_synthetic_needed > 0:
        print(f"🔄 正在補充 {num_synthetic_needed} 筆模擬數據...")
        for _ in range(num_synthetic_needed): roadmap.append('P' if random.random() < 0.4932 else 'B')
    print(f"✅ 數據準備完畢，總數據量: {len(roadmap)}。")
    
    roadmap_numeric = np.array([LABEL_MAP[r] for r in roadmap if r in LABEL_MAP]).reshape(-1, 1)

    # --- 1. 訓練 HMM 專家 ---
    print("\n--- [開始訓練 HMM 專家] ---")
    hmm_model = hmm.CategoricalHMM(n_components=2, n_iter=200, random_state=42, tol=1e-3, init_params="ste")
    hmm_model.fit(roadmap_numeric)
    joblib.dump(hmm_model, os.path.join(MODEL_DIR, 'hmm_model.pkl'))
    print("✅ HMM 專家 (hmm_model.pkl) 已儲存。")

    # --- 2. 訓練 XGBoost 專家 ---
    print("\n--- [開始訓練進階 XGBoost 專家] ---")
    X_xgb, y_xgb = extract_xgb_features(roadmap)
    if len(X_xgb) == 0: print("❌ XGBoost 特徵提取失敗"); return
    X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb)
    scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("✅ XGB 標準化器 (scaler.pkl) 已儲存。")
    xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=200, learning_rate=0.05, max_depth=5, use_label_encoder=False, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    accuracy = xgb_model.score(X_test_scaled, y_test)
    print(f"📈 XGBoost 準確率: {accuracy:.4f}")
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    print("✅ XGBoost 專家 (xgb_model.pkl) 已儲存。")
    
    # --- 3. 訓練 LSTM 專家 ---
    print("\n--- [開始訓練 LSTM 深度記憶專家] ---")
    X_lstm, y_lstm = prepare_lstm_data(roadmap_numeric.flatten())
    if len(X_lstm) == 0: print("❌ LSTM 數據準備失敗"); return
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42, stratify=y_lstm)
    
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LSTM_SEQUENCE_LENGTH, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("🔄 正在訓練 LSTM 模型...")
    lstm_model.fit(X_train_l, y_train_l, epochs=100, batch_size=64, validation_split=0.1, callbacks=[early_stopping], verbose=0)
    loss, accuracy = lstm_model.evaluate(X_test_l, y_test_l, verbose=0)
    print(f"📈 LSTM 準確率: {accuracy:.4f}")
    lstm_model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    print("✅ LSTM 專家 (lstm_model.h5) 已儲存。")

    print("\n🎉 所有三體AI專家模型已成功訓練並儲存！")

if __name__ == '__main__':
    train()

