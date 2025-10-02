import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# --- 檔案與欄位定義 (ETP 數據) ---
INPUT_FILE = 'melbourne_etp_optimized_training_data.csv' 
TARGET_COL = 'Energy Consumption' # 總能耗，迴歸目標

# 預先定義輸入特徵 (X)
# 排除目標和創建的 Pass 標籤，以及水質指標本身（它們是約束條件）
EXCLUDE_COLS = [
    TARGET_COL, 'Quality_Pass', 'BOD_Pass', 'COD_Pass', 
    'Biological Oxygen Demand', 'Chemical Oxygen Demand'
]

print("--- 專案 3：墨爾本 ETP 能耗優化 (XGBoost 迴歸) ---")

try:
    # 1. 載入優化後的數據集
    df = pd.read_csv(INPUT_FILE, index_col=0)
    
    # 2. 定義 X 和 Y
    X_features = [col for col in df.columns if col not in EXCLUDE_COLS]
    
    X = df[X_features].astype(float)
    Y = df[TARGET_COL].astype(float)

    print(f"模型將使用 {len(X_features)} 個特徵進行訓練: {X_features}")

    # 3. 劃分訓練集和測試集 (90/10)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    # 4. 訓練 XGBoost 迴歸模型 (調整參數以適應小樣本)
    regressor = XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42,
        n_jobs=-1
    )
    
    print("⏳ 開始訓練 XGBoost 迴歸模型...")
    regressor.fit(X_train, Y_train)
    print("✅ 模型訓練完成！")

    # 5. 模型預測與評估
    Y_pred = regressor.predict(X_test)
    
    # 使用 np.sqrt 手動計算 RMSE
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    # 使用 np.mean 手動計算 MAE
    mae = np.mean(np.abs(Y_test - Y_pred)) 

    print("\n--- 模型性能評估 (目標: 總能耗 kWh) ---")
    print(f"RMSE (均方根誤差): {rmse:.2f} kWh")
    print(f"MAE (平均絕對誤差): {mae:.2f} kWh")
    print(f"平均能耗: {Y_test.mean():.2f} kWh")
    print(f"MAE 佔平均能耗的百分比: {(mae / Y_test.mean() * 100):.2f}%")
    
    # 6. SHAP 可解釋性分析
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test)

    # 找出並打印前三個重要特徵
    feature_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns)
    top_3_features = feature_importance.nlargest(3)

    print("\n--- 前 3 個最重要的能耗影響特徵 (SHAP 值) ---")
    for name, value in top_3_features.items():
        print(f"  - {name}: {value:.2f}")

    # 7. 繪製 SHAP 摘要圖
    print("生成 SHAP 摘要圖，請查看顯示視窗...")
    shap.summary_plot(shap_values, X_test, show=True)
    plt.title("SHAP Feature Importance for ETP Energy Prediction (Optimized Data)")
    plt.tight_layout()

except FileNotFoundError:
    print(f"🚨 錯誤：找不到檔案 {INPUT_FILE}。請確保您已成功運行數據準備腳本。")
except Exception as e:
    print(f"🚨 運行時發生錯誤: {e}")

    import joblib 

# 儲存訓練好的 XGBoost 模型
MODEL_FILE = 'etp_energy_model.joblib'
joblib.dump(regressor, MODEL_FILE)
print(f"✅ 模型已儲存至：{MODEL_FILE}")

# 儲存特徵列表 (確保 Web App 使用相同的順序)
FEATURE_LIST_FILE = 'etp_features.joblib'
joblib.dump(X_features, FEATURE_LIST_FILE)
print(f"✅ 特徵列表已儲存至：{FEATURE_LIST_FILE}")