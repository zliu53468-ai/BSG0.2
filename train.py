# -*- coding: utf-8 -*-
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from hmmlearn import hmm
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 全域設定
# =============================================================================
MODEL_DIR = 'models'
N_FEATURES_WINDOW = 20
LABEL_MAP = {'B': 0, 'P': 1}
REVERSE_MAP = {0: 'B', 1: 'P'}  # 預測時使用英文代碼，與前端保持一致

# 真實歷史數據
REAL_HISTORY_DATA = [
    "P", "P", "T", "B", "T", "B", "P", "B", "P", "P", "B", "B", "T", "B", "B", "P", "B", "B", "P", "B", "B", "T", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "P", "P", "B", "P", "B", "P", "T", "P", "B", "P", "B", "P", "T", "T", "B", "P", "B", "B", "P", "B", "B", "P", "T", "T", "B", "P", "B", "B", "B", "B", "B", "P", "P", "B", "P", "B", "B", "P", "P", "P", "P", "P", "P", "B", "B", "T", "B", "T", "B", "P", "P", "P", "B", "P", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "T", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "T", "T", "B", "B", "B", "B", "P", "B", "B", "B", "P", "B", "T", "P", "P", "B", "B", "B", "P", "P", "P", "B", "P", "B", "P", "P", "P", "B", "T", "B", "P", "B", "T", "B", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "B", "T", "T", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "T", "B", "P", "B", "P", "B", "T", "P", "B", "B", "P", "B", "B", "P", "T", "B", "P", "T", "B", "B", "B", "P", "T", "B", "B", "P", "B", "B", "P", "T", "B", "B", "P", "B", "P", "B", "T", "B", "B", "P", "P", "B", "B", "P", "T", "P", "P", "B", "P", "B", "B", "B", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "T", "T", "B", "P", "P", "B", "P", "P", "B", "B", "P", "B", "P", "T", "P", "P", "P", "P", "B", "B", "B", "B", "B", "P", "B", "P", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "P", "B", "P", "B", "P", "T", "B", "P", "B", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "B", "T", "B", "T", "B", "P", "B", "P", "T", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "B", "B", "T", "P", "B", "P", "B", "P", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "P", "P", "B", "P", "P", "P", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "B", "P", "P", "T", "P", "B", "B", "T", "B", "P", "T", "P", "B", "P", "B", "B", "P", "B", "B", "T", "P", "P", "P", "P", "T", "P", "T", "B", "B", "P", "B", "B", "P", "P", "P", "B", "P", "B", "P", "T", "P", "P", "T", "P", "P", "B", "P", "P", "B", "P", "P", "B", "P", "P", "T", "B", "P", "B", "P", "P", "B", "B", "B", "B", "T", "T", "T", "B", "B", "B", "B", "B", "B", "P", "P", "P", "T", "P", "T", "B", "P", "P", "T", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "B", "B", "B", "P", "P", "P", "P", "P", "T", "P", "B", "P", "P", "B", "T", "B", "B", "B", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "P", "B", "P", "P", "B", "P", "B", "B", "P", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "T", "T", "B", "P", "B", "B", "B", "T", "T", "B", "B", "P", "B", "T", "P", "B", "P", "B", "P", "P", "P", "B", "P", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "T", "T", "B", "B", "P", "B", "B", "P", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "B", "P", "P", "B", "T", "B", "B", "P", "P", "P", "T", "P", "B", "T", "P", "B", "B", "P", "B", "B", "B", "B", "B", "P", "B", "T", "T", "P", "B", "B", "B", "P", "B", "B", "P", "B", "P", "B", "P", "P", "P", "P", "P", "P", "B", "B", "B", "P", "T", "P", "B", "T", "B", "B", "B", "B", "T", "B", "P", "B", "B", "B", "B", "B", "B", "P", "B", "P", "B", "B", "P", "P", "B", "P", "P", "P", "P", "B", "B", "B", "B", "B", "T", "B", "B", "P", "B", "P", "T", "P", "B", "B", "P", "B", "B", "B", "P", "P", "P", "B", "P", "P", "B", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "B", "B", "B", "B", "B", "B", "P", "T", "P", "B", "P", "B", "P", "P", "B", "B", "P", "B", "P", "P", "T", "B", "B", "P", "P", "B", "B", "P", "B", "B", "T", "P", "P", "B", "T", "P", "B", "B", "P", "B", "P", "B", "P", "B", "B", "B", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "T", "P", "P", "B", "T", "B", "P", "P", "P", "B", "B", "P", "B", "B", "P", "B", "P", "P", "B", "B", "B", "B", "P", "P", "T", "B", "B", "P", "P", "B", "P", "B", "P", "P", "P", "P", "B", "B", "P", "P", "B", "P", "P", "T", "P", "P", "P", "B", "B", "P", "P", "T", "P", "B", "P", "B", "B", "P", "P", "P", "B", "B", "P", "P", "B", "P", "T", "P", "P", "P", "B", "B", "P", "P", "B", "P", "B", "B", "P", "T", "B", "P", "T", "T", "P", "T", "B", "T", "P", "T", "P", "T", "P", "P", "B", "B", "P", "P", "P", "P", "P"
]

# =============================================================================
# 路單分析核心 (BaccaratAnalyzer) - 保持不變
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
# 特徵工程與訓練 - 重點修改部分
# =============================================================================
def extract_features(full_roadmap):
    features_list, labels = [], []
    
    # 確保有足夠的數據點來創建特徵和標籤
    if len(full_roadmap) <= N_FEATURES_WINDOW:
        return np.array([]), np.array([])
    
    # 從 N_FEATURES_WINDOW 開始，因為我們需要至少N個數據點來構建特徵
    for i in range(N_FEATURES_WINDOW, len(full_roadmap)):
        # 關鍵修改：只使用歷史數據來構建特徵 (0到i-1)
        historical_roadmap = full_roadmap[:i]
        # 標籤是當前數據點 (i)
        label = full_roadmap[i]
        
        if label not in LABEL_MAP:
            continue
            
        # 從歷史數據中提取最後N個結果作為窗口
        window = historical_roadmap[-N_FEATURES_WINDOW:]
        
        # 計算基本特徵
        b_count = window.count('B')
        p_count = window.count('P')
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
        
        basic_features = [b_ratio, p_ratio, streak, streak_type, prev_result]
        
        # 使用歷史數據創建分析器
        analyzer = BaccaratAnalyzer(historical_roadmap)
        derived_features = analyzer.get_derived_road_features()
        
        # 合併所有特徵
        all_features = basic_features + derived_features
        features_list.append(all_features)
        labels.append(LABEL_MAP[label])
        
    return np.array(features_list), np.array(labels)

def train():
    print("="*50)
    print("開始重新訓練 AI 模型 (使用正確的時間序列方法)...")
    print("="*50)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"✅ 已建立目錄: {MODEL_DIR}")
    
    # 只使用真實數據，不再生成合成數據
    roadmap = [r for r in REAL_HISTORY_DATA if r in ['B', 'P']]
    print(f"✅ 使用 {len(roadmap)} 筆真實歷史數據進行訓練")
    
    # 轉換為數值格式供HMM使用
    roadmap_numeric = np.array([LABEL_MAP[r] for r in roadmap]).reshape(-1, 1)
    
    # --- 1. 訓練 HMM 專家 ---
    print("\n--- [開始訓練 HMM 專家] ---")
    try:
        hmm_model = hmm.CategoricalHMM(
            n_components=2, 
            n_iter=200, 
            random_state=42, 
            tol=1e-3, 
            init_params="ste"
        )
        hmm_model.fit(roadmap_numeric)
        joblib.dump(hmm_model, os.path.join(MODEL_DIR, 'hmm_model.pkl'))
        print("✅ HMM 專家 (hmm_model.pkl) 已儲存。")
    except Exception as e:
        print(f"❌ HMM 訓練失敗: {e}")
        return
    
    # --- 2. 訓練 XGBoost & LightGBM 專家 ---
    print("\n--- [開始訓練 XGBoost & LightGBM 專家] ---")
    
    # 提取特徵和標籤
    X, y = extract_features(roadmap)
    if len(X) == 0:
        print("❌ 特徵提取失敗 - 數據不足")
        return
        
    print(f"✅ 成功提取 {X.shape[0]} 個樣本，每個樣本有 {X.shape[1]} 個特徵")
    
    # 使用時間序列交叉驗證評估模型
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_scores, lgbm_scores = [], []
    
    # 創建標準化器並擬合全部數據
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("✅ 標準化器 (scaler.pkl) 已儲存。")
    
    print("\n進行時間序列交叉驗證...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 訓練XGBoost
        xgb_model = XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss', 
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=3,  # 減少深度防止過擬合
            use_label_encoder=False, 
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_score = xgb_model.score(X_test, y_test)
        xgb_scores.append(xgb_score)
        
        # 訓練LightGBM
        lgbm_model = lgb.LGBMClassifier(
            objective='binary', 
            metric='binary_logloss', 
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=3,  # 減少深度防止過擬合
            random_state=42
        )
        lgbm_model.fit(X_train, y_train)
        lgbm_score = lgbm_model.score(X_test, y_test)
        lgbm_scores.append(lgbm_score)
        
        print(f"折疊 {fold+1}: XGBoost={xgb_score:.4f}, LightGBM={lgbm_score:.4f}")
    
    # 輸出交叉驗證結果
    print(f"\n📊 XGBoost 平均準確率: {np.mean(xgb_scores):.4f} (±{np.std(xgb_scores):.4f})")
    print(f"📊 LightGBM 平均準確率: {np.mean(lgbm_scores):.4f} (±{np.std(lgbm_scores):.4f})")
    
    # 使用全部數據訓練最終模型
    print("\n使用全部數據訓練最終模型...")
    
    # XGBoost
    xgb_model = XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss', 
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=3,
        use_label_encoder=False, 
        random_state=42
    )
    xgb_model.fit(X_scaled, y)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    print("✅ XGBoost 專家 (xgb_model.pkl) 已儲存。")
    
    # LightGBM
    lgbm_model = lgb.LGBMClassifier(
        objective='binary', 
        metric='binary_logloss', 
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=3,
        random_state=42
    )
    lgbm_model.fit(X_scaled, y)
    joblib.dump(lgbm_model, os.path.join(MODEL_DIR, 'lgbm_model.pkl'))
    print("✅ LightGBM 專家 (lgbm_model.pkl) 已儲存。")
    
    # 輸出最終模型在訓練集上的表現
    xgb_train_score = xgb_model.score(X_scaled, y)
    lgbm_train_score = lgbm_model.score(X_scaled, y)
    print(f"\n🎯 最終模型訓練集準確率:")
    print(f"   XGBoost: {xgb_train_score:.4f}")
    print(f"   LightGBM: {lgbm_train_score:.4f}")
    
    # 輸出分類報告
    y_pred_xgb = xgb_model.predict(X_scaled)
    y_pred_lgbm = lgbm_model.predict(X_scaled)
    
    print("\n📋 XGBoost 分類報告:")
    print(classification_report(y, y_pred_xgb, target_names=['莊(B)', '閒(P)']))
    
    print("📋 LightGBM 分類報告:")
    print(classification_report(y, y_pred_lgbm, target_names=['莊(B)', '閒(P)']))
    
    print("\n🎉 所有專家模型已成功訓練並儲存！")

if __name__ == '__main__':
    train()
