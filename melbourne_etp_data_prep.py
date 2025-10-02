import pandas as pd
import numpy as np
import os

# --- 檔案與欄位定義 (ETP 數據) ---
DATA_FILE = 'melbourne_etp_data.csv' 
TARGET_COL = 'Energy Consumption'  
EFFLUENT_BOD_COL = 'Biological Oxygen Demand'
EFFLUENT_COD_COL = 'Chemical Oxygen Demand'

# --- 主清洗與標籤創建流程 ---
if __name__ == "__main__":
    
    print("--- 1. 載入 ETP 數據並清洗 ---")
    try:
        # 載入數據，並將第一欄設為索引 (假設它是 time-related index)
        df = pd.read_csv(DATA_FILE, index_col=0)
        
        # 處理缺失值和非數值數據
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 簡單的缺失值處理：向前填充 (ffill) 再向後填充 (bfill)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 必須先進行一次dropna，確保基本數據沒有缺失
        df = df.dropna()

        print(f"✅ 數據載入與清洗成功。總記錄數：{len(df)}")
        
    except Exception as e:
        print(f"🚨 載入或清洗時發生錯誤: {e}")
        exit()

     # --- 2. 進行特徵工程 (只保留最有效的滯後指標) ---

    # 🚨 刪除 Ammonia_Daily_Change
    # df['Ammonia_Daily_Change'] = df['Ammonia'].diff()

    # 🚨 刪除 Inflow_Daily_Change
    # df['Inflow_Daily_Change'] = df['Average Inflow'].diff()

    # 特徵 3: 氨氮的 1 天滯後值 (保留)
    df['Ammonia_Lag_1Day'] = df['Ammonia'].shift(1)
    
    # 清理特徵工程產生出來的 NaN (diff/shift 的第一筆會是 NaN)
    # 這些 NaN 會刪除少數記錄，但確保了訓練數據的完整性
    df = df.dropna()
    print(f"✅ 特徵工程完成。新增 3 個欄位。剩餘記錄數: {len(df)}")

    # --- 3. 創建水質達標標籤 (約束條件 Quality_Pass) ---
    BOD_THRESHOLD = 300  # mg/L
    COD_THRESHOLD = 700  # mg/L

    print("\n--- 3. 創建 Quality_Pass 達標標籤 (約束條件) ---")

    df['BOD_Pass'] = (df[EFFLUENT_BOD_COL] < BOD_THRESHOLD).astype(int)
    df['COD_Pass'] = (df[EFFLUENT_COD_COL] < COD_THRESHOLD).astype(int)
    df['Quality_Pass'] = df['BOD_Pass'] * df['COD_Pass']

    pass_rate = df['Quality_Pass'].mean() * 100

    print(f"✅ 'Quality_Pass' 欄位已創建。總記錄的達標率: {pass_rate:.2f}%")

    # --- 4. 創建優化訓練集並儲存 ---
    df_optimized = df[df['Quality_Pass'] == 1].copy()

    OUTPUT_FILE = 'melbourne_etp_optimized_training_data.csv'
    df_optimized.to_csv(OUTPUT_FILE, index=True)

    print(f"\n✅ 已將 {len(df_optimized)} 筆達標記錄儲存至：{OUTPUT_FILE}")