import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- 載入模型和常數 ---
MODEL_FILE = 'etp_energy_model_lag.joblib'
FEATURE_LIST_FILE = 'etp_features_lag.joblib'
DATA_SAMPLE_FILE = 'melbourne_etp_optimized_lag_training_data.csv'

try:
    # 載入模型、特徵和用於 SHAP/敏感度分析的數據樣本
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURE_LIST_FILE)
    df_optimized = pd.read_csv(DATA_SAMPLE_FILE, index_col=0)
    
    # 這裡需要您運行新模型後的結果，暫時使用舊結果
    AVG_ENERGY = 270716.17  
    MAE = 32143.43 
    
    # 為 SHAP 繪圖準備數據
    X_sample = df_optimized[features].astype(float)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    IS_LOADED = True
except FileNotFoundError:
    IS_LOADED = False
    # ... (錯誤處理) ...

# --- Streamlit 介面設定 ---
st.set_page_config(layout="wide", page_title="ETP 能耗優化決策系統 V2")
st.title("🌊 ETP 能耗優化決策系統 V2")
st.markdown("### 整合時間序列的預測與決策支援")

# ... (如果載入失敗則停止) ...

# --- 側邊欄輸入控制 ---
st.sidebar.header("⚙️ 實時輸入與控制變數")

# 1. 調整能耗驅動因素
st.sidebar.subheader("影響最大的 3 個因素:")
temp_input = st.sidebar.slider('平均溫度 (°C)', min_value=5.0, max_value=35.0, value=20.0, step=0.1)
ammonia_input = st.sidebar.slider('當前 Ammonia (mg/L)', min_value=10.0, max_value=60.0, value=40.0, step=1.0)
inflow_input = st.sidebar.slider('當前 Inflow (MLD)', min_value=2.0, max_value=8.0, value=4.5, step=0.1)

# 2. 滯後特徵 (假設輸入歷史值)
st.sidebar.subheader("歷史負荷條件 (Lag Features):")
ammonia_lag1 = st.sidebar.slider('昨日 Ammonia (mg/L)', min_value=10.0, max_value=60.0, value=35.0, step=1.0)
inflow_lag1 = st.sidebar.slider('昨日 Inflow (MLD)', min_value=2.0, max_value=8.0, value=4.0, step=0.1)

# ... (建立輸入 DataFrame) ...

# --- 主面板佈局 ---
tab1, tab2, tab3 = st.tabs(["預測與優化建議", "SHAP 模型解釋", "敏感度分析"])

# === Tab 1: 預測與優化建議 (與舊版類似，但使用新模型) ===
with tab1:
    # 運行預測並顯示結果 (這裡省略了完整的預測程式碼，請確保您的輸入 DataFrame 順序正確)
    st.header("功能開發中，請先查看模型解釋與敏感度分析。")

# === Tab 2: SHAP 模型解釋 (整合 SHAP 視覺化) ===
with tab2:
    st.header("模型可解釋性：哪些因素在驅動能耗？")
    st.markdown("該圖顯示了所有達標記錄中，每個特徵對能耗的影響程度和方向。")
    
    # 1. SHAP Summary Plot
    st.subheader("1. SHAP 特徵重要性總結")
    fig, ax = plt.subplots(figsize=(10, 6))
    # 創建 SHAP 摘要圖
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
    st.pyplot(fig)
    st.info("紅點表示特徵值高，藍點表示特徵值低。紅點在右側表示該特徵值高時會推高能耗。")

    # 2. SHAP Force Plot (第一個樣本的詳細解釋)
    st.subheader("2. 單一樣本預測詳情 (Force Plot)")
    st.markdown("請在瀏覽器中開啟 **etp_energy_force_plot.html** 檔案，可獲得更佳的互動體驗。")

# === Tab 3: 敏感度分析 (新增功能) ===
with tab3:
    st.header("敏感度分析：量化變數變動對能耗的影響")
    st.markdown("此分析模擬當某個關鍵特徵變化時，能耗預測值的變化趨勢。")
    
    sensitivity_feature = st.selectbox(
        '選擇要分析的變數', 
        options=['Average Temperature', 'Ammonia', 'Average Inflow'], 
        index=0
    )
    
    # 執行敏感度分析
    base_prediction = model.predict(X_sample.iloc[0:1].copy())[0]
    
    # 創建變動範圍 (例如在平均值 ± 20% 範圍內)
    if sensitivity_feature in X_sample.columns:
        base_value = X_sample[sensitivity_feature].mean()
        min_val = base_value * 0.8
        max_val = base_value * 1.2
        test_values = np.linspace(min_val, max_val, 20)
        
        predictions = []
        
        for val in test_values:
            temp_input = X_sample.iloc[0:1].copy()
            temp_input[sensitivity_feature] = val
            pred = model.predict(temp_input)[0]
            predictions.append(pred)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_values, predictions, marker='o', linestyle='-', color='b')
        ax.set_title(f'{sensitivity_feature} 變化對能耗的影響')
        ax.set_xlabel(f'{sensitivity_feature} 值')
        ax.set_ylabel('預測能耗 (kWh)')
        st.pyplot(fig)
    else:
        st.warning(f"特徵 {sensitivity_feature} 尚未納入模型。")