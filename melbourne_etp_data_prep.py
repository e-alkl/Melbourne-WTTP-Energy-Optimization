import pandas as pd
import numpy as np
import os

# --- 檔案與欄位定義 ---
DATA_FILE = 'melbourne_etp_data.csv' 
TARGET_COL = 'Energy Consumption'  

# 假設的關鍵操作控制變數 (如果您的數據集有，請確認欄位名)
# 由於我們沒有實際的操作數據，我們暫時使用 'Average Outflow' 作為一個操作代理指標，
# 但最佳實踐應使用 DO_SETPOINT。這裡我們仍然保持 Energy 為 Y。
# NEW_CONTROL_COL = 'DO_SETPOINT' # 假設您有這個欄位，如果沒有，請忽略。

EFFLUENT_BOD_COL = 'Biological Oxygen Demand'
EFFLUENT_COD_COL = 'Chemical Oxygen Demand'   

# --- Lag Feature 設置 ---
LAG_FEATURES = {
    'Ammonia': [1, 2],         # 增加前 1 天和前 2 天的 Ammonia
    'Average Inflow': [1, 2],  # 增加前 1 天和前 2 天的 Inflow
}

# --- 創建滯後特徵的函式 ---
def create_lag_features(df, column, lags):
    """為指定欄位創建滯後特徵 (Lag Features)。"""
    for lag in lags:
        new_col = f'{column}_Lag_{lag}'
        # shift(1) 是前一天的數據，shift(2) 是前兩天的數據 (假設是日數據)
        df[new_col] = df[column].shift(lag)
    return df

# --- 主清洗、特徵工程與標籤創建流程 ---
if __name__ == "__main__":
    print("--- 1. 載入 ETP 數據、清洗並執行特徵工程 ---")
    try:
        df = pd.read_csv(DATA_FILE, index_col=0)
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 排序時間索引 (關鍵，確保 Lag Feature 正確)
        df = df.sort_index()

        # 執行滯後特徵工程
        for col, lags in LAG_FEATURES.items():
            if col in df.columns:
                df = create_lag_features(df, col, lags)
                print(f"   - 成功為 {col} 創建 Lag Features。")
            else:
                print(f"   - 警告：未找到欄位 {col} 來創建 Lag Features。")

        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        print(f"✅ 數據載入、清洗與特徵工程成功。總記錄數：{len(df)}")
        
    except Exception as e:
        print(f"🚨 載入或清洗時發生錯誤: {e}")
        exit()

    # --- 2. 創建水質達標標籤 (使用之前調整的門檻) ---
    BOD_THRESHOLD = 300  # mg/L
    COD_THRESHOLD = 700  # mg/L

    df['BOD_Pass'] = (df[EFFLUENT_BOD_COL] < BOD_THRESHOLD).astype(int)
    df['COD_Pass'] = (df[EFFLUENT_COD_COL] < COD_THRESHOLD).astype(int)
    df['Quality_Pass'] = df['BOD_Pass'] * df['COD_Pass']
    pass_rate = df['Quality_Pass'].mean() * 100

    print(f"\n✅ 達標率 (新特徵已包含): {pass_rate:.2f}%")

    # --- 3. 創建優化訓練集並儲存 ---
    df_optimized = df[df['Quality_Pass'] == 1].copy()

    OUTPUT_FILE = 'melbourne_etp_optimized_lag_training_data.csv'
    df_optimized.to_csv(OUTPUT_FILE, index=True)

    print(f"\n✅ 已將 {len(df_optimized)} 筆達標記錄 (含 Lag Features) 儲存至：{OUTPUT_FILE}")