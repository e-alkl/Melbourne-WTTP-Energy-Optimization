import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 檔案與欄位定義 ---
INPUT_FILE = 'melbourne_etp_optimized_lag_training_data.csv' 
TARGET_COL = 'Energy Consumption'

MODEL_FILE = 'etp_energy_model_lag.joblib'
FEATURE_LIST_FILE = 'etp_features_lag.joblib'

# 排除清單：目標變數、約束標籤，以及被 Lag Features 替換的原始變數
EXCLUDE_COLS = [
    TARGET_COL, 'Quality_Pass', 'BOD_Pass', 'COD_Pass', 
    'Biological Oxygen Demand', 'Chemical Oxygen Demand',
    # 排除原始 Ammonia 和 Inflow，強迫模型使用帶有時間上下文的 Lag Features
    'Ammonia', 'Average Inflow' 
]

if __name__ == "__main__":
    print("--- 階段 1：訓練帶有 Lag Features 的 XGBoost 模型 ---")
    
    try:
        df = pd.read_csv(INPUT_FILE, index_col=0)
        
        # 1. 定義 X 和 Y
        X_features = [col for col in df.columns if col not in EXCLUDE_COLS]
        
        X = df[X_features].astype(float)
        Y = df[TARGET_COL].astype(float)

        print(f"模型將使用 {len(X_features)} 個特徵 (含 Lag Features) 進行訓練。")

        # 2. 劃分數據
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )

        # 3. 訓練模型
        regressor = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, n_jobs=-1)
        print("⏳ 開始 XGBoost 訓練...")
        regressor.fit(X_train, Y_train)
        print("✅ 模型訓練完成！")

        # 4. 評估
        Y_pred = regressor.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        mae = mean_absolute_error(Y_test, Y_pred)

        print("\n--- 模型性能評估 (含 Lag Features) ---")
        print(f"RMSE: {rmse:.2f} kWh")
        print(f"MAE: {mae:.2f} kWh")
        print(f"平均能耗: {Y_test.mean():.2f} kWh")
        print(f"MAE 佔比: {(mae / Y_test.mean() * 100):.2f}%")
        
        # 5. 儲存模型和特徵
        joblib.dump(regressor, MODEL_FILE)
        joblib.dump(X_features, FEATURE_LIST_FILE)
        print(f"\n✅ 模型已儲存至：{MODEL_FILE}")
        
    except FileNotFoundError:
        print(f"🚨 錯誤：找不到輸入檔案 {INPUT_FILE}。請先運行 'melbourne_etp_data_prep.py' 腳本。")
    except Exception as e:
        print(f"🚨 訓練時發生錯誤: {e}")