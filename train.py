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

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# =============================================================================
# 全域設定
# =============================================================================
MODEL_DIR = 'models'
TOTAL_DATA_SIZE = 8000
N_FEATURES_WINDOW = 20
LABEL_MAP = {'B': 0, 'P': 1}
DERIVED_ROAD_MAP = {'R': 1, 'B': 0} # 紅=1, 藍=0

REAL_HISTORY_DATA = [
    "P", "P", "T", "B", "T", "B", "P", "B", "P", "P", "B", "B", "T", "B", "B", "P", "B", "B", "P", "B", "B", "T", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "P", "B", "P", "T", "T", "B", "P", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "B", "B", "B", "P", "P", "B", "P", "B", "B", "P", "P", "P", "P", "P", "P", "B", "B", "T", "B", "T", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "T", "T", "B", "B", "B", "B", "P", "B", "B", "B", "P", "B", "T", "P", "P", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "P", "P", "B", "T", "B", "P", "B", "T", "B", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "B", "T", "T", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "T", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "P", "T", "B", "P", "T", "B", "B", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "B", "B", "P", "B", "P", "B", "T", "B", "B", "P", "P", "B", "B", "P", "T", "P", "P", "B", "P", "B", "B", "B", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "T", "T", "B", "P", "P", "B", "P", "P", "B", "B", "P", "B", "P", "T", "P", "P", "P", "P", "B", "B", "B", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "P", "B", "P", "B", "P", "T", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "B", "T", "B", "T", "B", "P", "B", "P", "T", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "P", "P", "B", "P", "P", "P", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "B", "P", "P", "T", "P", "B", "B", "T", "B", "P", "T", "P", "B", "P", "B", "B", "P", "B", "B", "T", "P", "P", "P", "P", "T", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "T", "P", "P", "T", "P", "P", "B", "P", "P", "B", "P", "P", "B", "P", "P", "T", "B", "P", "B", "P", "P", "B", "B", "B", "B", "T", "T", "T", "B", "B", "B", "B", "B", "B", "P", "P", "P", "T", "P", "T", "B", "P", "P", "T", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "P", "P", "T", "P", "B", "P", "P", "B", "T", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "T", "T", "B", "P", "B", "B", "B", "T", "T", "B", "B", "P", "B", "T", "P", "B", "P", "B", "P", "P", "P", "B", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", "B", "B", "P", "B", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "B", "B", "P", "B", "T", "T", "P", "B", "B", "B", "P", "B", "B", "P", "B", "P", "B", "P", "P", "P", "P", "P", "P", "B", "B", "B", "P", "T", "P", "B", "T", "B", "B", "B", "B", "T", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "B", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "B", "B", "B", "T", "B", "B", "P", "B", "P", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "P", "B", "P", "P", "B", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "B", "B", "B", "B", "B", "B", "P", "T", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "P", "P", "T", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "P", "B", "T", "P", "B", "B", "P", "B", "P", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "T", "P", "P", "B", "T", "B", "P", "P", "P", "B", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "B", "P", "P", "T", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "P", "T", "P", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "T", "T", "P", "T", "B", "T", "P", "T", "P", "T", "P", "P", "B", "B", "P", "P", "P", "P", "P"
]

# =============================================================================
# 路單分析核心 (BaccaratAnalyzer)
# =============================================================================
class BaccaratAnalyzer:
    def __init__(self, roadmap):
        self.roadmap = [r for r in roadmap if r in ['B', 'P', 'T']]
        self.big_road_grid = self._generate_big_road_grid()

    def _generate_big_road_grid(self):
        grid = []
        if not self.roadmap:
            return grid
        
        filtered_map = [r for r in self.roadmap if r in ['B', 'P']]
        if not filtered_map:
            return grid

        current_col = []
        last_result = None
        for result in filtered_map:
            if result != last_result and last_result is not None:
                grid.append(current_col)
                current_col = []
            current_col.append(result)
            last_result = result
        if current_col: grid.append(current_col)
        return grid

    def _get_derived_bead(self, c, r, offset):
        if c - offset < 0: return None 
        
        len_col_offset = len(self.big_road_grid[c - offset])
        len_col_prev = len(self.big_road_grid[c - 1])

        # 比較點 (c-offset, r) vs (c-offset, r-1)
        bead_at_pos = self.big_road_grid[c-offset][r] if r < len_col_offset else None
        bead_above = self.big_road_grid[c-offset][r-1] if r > 0 and r-1 < len_col_offset else None

        if bead_at_pos:
            if bead_above:
                return 'R' # 直落, 紅
            else:
                return 'B' # 換列第一個, 藍
        else:
            if bead_above:
                 # (c-offset,r) 為空, 但 (c-offset,r-1) 存在, 代表到底了
                 # 此時比較 c-1 和 c-1-offset 的長度 (齊整)
                len_col_prev_offset = len(self.big_road_grid[c-1-offset]) if c-1-offset >= 0 else 0
                return 'R' if len_col_prev == len_col_prev_offset else 'B'
            else:
                return 'B' # 藍, 應該不會發生

    def get_derived_roads(self):
        roads = {'big_eye': [], 'small': [], 'cockroach': []}
        if len(self.big_road_grid) < 2: return roads

        offsets = {'big_eye': 1, 'small': 2, 'cockroach': 3}

        for name, offset in offsets.items():
            road_data = []
            start_col = offset + 1
            if len(self.big_road_grid) < start_col: continue

            for c in range(start_col - 1, len(self.big_road_grid)):
                col_data = []
                start_row = 1 if c == (start_col - 1) and len(self.big_road_grid[c - 1]) == 1 and len(self.big_road_grid[c-offset]) > 1 else 0

                for r in range(start_row, len(self.big_road_grid[c])):
                    bead = self._get_derived_bead(c, r, offset)
                    if bead: 
                        col_data.append(bead)
                if col_data:
                    # 根據標準畫法，如果第一顆是藍色，前面補一顆紅色
                    if len(road_data) == 0 and col_data[0] == 'B':
                        road_data.append(['R'])
                    road_data.append(col_data)
            roads[name] = road_data
        return roads
        
    def get_derived_road_predictions(self):
        predictions = {'big_eye': 0.5, 'small': 0.5, 'cockroach': 0.5}
        roads_data = self.get_derived_roads()
        
        for name, road in roads_data.items():
            if not road: continue
            
            flat_road = [bead for col in road for bead in col]
            if len(flat_road) > 2:
                window = flat_road[-10:]
                red_count = window.count('R')
                predictions[name] = red_count / len(window) if window else 0.5
        
        return [predictions['big_eye'], predictions['small'], predictions['cockroach']]

# =============================================================================
# 特徵工程與訓練
# =============================================================================
def extract_full_features(full_roadmap):
    features_list = []
    labels = []
    if len(full_roadmap) <= N_FEATURES_WINDOW:
        return np.array([]), np.array([])

    for i in range(N_FEATURES_WINDOW, len(full_roadmap)):
        current_roadmap = full_roadmap[:i]
        label = full_roadmap[i]
        if label not in LABEL_MAP: continue

        window = current_roadmap[-N_FEATURES_WINDOW:]
        b_count = window.count('B'); p_count = window.count('P'); total = b_count + p_count
        b_ratio = b_count / total if total > 0 else 0.5
        p_ratio = p_count / total if total > 0 else 0.5
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
        derived_features = analyzer.get_derived_road_predictions()
        
        all_features = basic_features + derived_features
        features_list.append(all_features)
        labels.append(LABEL_MAP[label])
        
    return np.array(features_list), np.array(labels)

def train():
    print("="*50)
    print("開始重新訓練 AI 模型 (含下三路進階特徵)...")
    print("="*50)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR); print(f"✅ 已建立目錄: {MODEL_DIR}")

    roadmap = list(REAL_HISTORY_DATA)
    print(f"🔄 已載入 {len(roadmap)} 筆真實歷史數據。")
    
    num_synthetic_needed = TOTAL_DATA_SIZE - len(roadmap)
    if num_synthetic_needed > 0:
        print(f"🔄 正在補充 {num_synthetic_needed} 筆模擬數據...")
        p_win_prob = 0.4932 
        for _ in range(num_synthetic_needed):
            roadmap.append('P' if random.random() < p_win_prob else 'B')
    print(f"✅ 數據準備完畢，總數據量: {len(roadmap)}。")

    print("\n--- [開始訓練 HMM 專家] ---")
    hmm_roadmap_numeric = np.array([LABEL_MAP[r] for r in roadmap if r in LABEL_MAP]).reshape(-1, 1)
    hmm_model = hmm.CategoricalHMM(n_components=2, n_iter=200, random_state=42, tol=1e-3, init_params="ste")
    hmm_model.fit(hmm_roadmap_numeric)
    joblib.dump(hmm_model, os.path.join(MODEL_DIR, 'hmm_model.pkl'))
    print("✅ HMM 專家 (hmm_model.pkl) 已儲存。")

    print("\n--- [開始訓練進階 XGBoost 專家] ---")
    print("🔄 正在提取基礎及下三路特徵...")
    X_full, y = extract_full_features(roadmap)
    if len(X_full) == 0:
        print("❌ 錯誤：無法提取任何特徵。訓練中止。")
        return
    print(f"✅ 特徵提取完成，共 {len(y)} 筆樣本，特徵維度: {X_full.shape[1]}。")

    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🔄 正在標準化特徵 (Scaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("✅ 標準化器 (scaler.pkl) 已儲存。")
    
    print("🔄 正在訓練 XGBoost 最終模型...")
    xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=200, learning_rate=0.05, max_depth=5, use_label_encoder=False, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    accuracy = xgb_model.score(X_test_scaled, y_test)
    print(f"📈 XGBoost在測試集上的準確率: {accuracy:.4f}")
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    print("✅ XGBoost 專家 (xgb_model.pkl) 已儲存。")
    
    print("\n🎉 所有專家模型已成功升級並儲存！")

if __name__ == '__main__':
    train()

