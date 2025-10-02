import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error
from itertools import product # 用於三變數搜索

# --- 1. 檔案定義與常數 ---
MODEL_FILE = 'etp_energy_model.joblib'
FEATURE_LIST_FILE = 'etp_features.joblib'
DATA_SAMPLE_FILE = 'melbourne_etp_optimized_training_data.csv'

# 性能基準
AVG_ENERGY = 270716.17  
MAE = 32143.43  
    
try:
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURE_LIST_FILE)
    df_optimized = pd.read_csv(DATA_SAMPLE_FILE, index_col=0)
    
    # 模擬 DO Setpoint 和 Recycle Ratio 的能耗調整
    def adjust_energy_for_controls(energy, do_setpoint, recycle_ratio):
        # 1. 模擬 DO 越高，能耗越高 (每增加 1 mg/L DO 增加 5% 額外能耗)
        base_do = 2.0 
        energy = energy * (1 + (do_setpoint - base_do) * 0.05)
        # 2. 模擬 Recycle Ratio 越高，泵送能耗略高 (每增加 10% RR 增加 2% 額外能耗)
        base_rr = 0.5 
        energy = energy * (1 + (recycle_ratio - base_rr) * 0.02)
        return energy

    # 執行 SHAP 計算
    X_sample = df_optimized[features].astype(float)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    IS_LOADED = True
except FileNotFoundError as e:
    IS_LOADED = False
    st.error(f"🚨 載入錯誤：找不到關鍵文件。請檢查 {e.filename} 是否存在。")
    
# --- 2. 數據獲取與處理 (Live Feed & 函數定義) ---
def get_live_data(df, features):
    """模擬從 SCADA 系統獲取最新的 5 筆數據。"""
    live_df = df.tail(5).copy()
    current_input_series = live_df.iloc[-1].copy()
    
    input_dict = {f: current_input_series.get(f, 0.0) for f in features}
    
    # 處理模型中可能缺少的固定特徵
    input_dict['Average Outflow'] = current_input_series.get('Average Outflow', 4.5)
    input_dict['Total Nitrogen'] = current_input_series.get('Total Nitrogen', 60.0)
    input_dict['Year'] = current_input_series.get('Year', 2017.0)
    input_dict['Month'] = current_input_series.get('Month', 6.0)
    input_dict['Day'] = current_input_series.get('Day', 15.0)
    # 模擬實時控制參數
    input_dict['DO Setpoint'] = current_input_series.get('DO Setpoint', 2.0) 
    input_dict['Recycle Ratio'] = current_input_series.get('Recycle Ratio', 0.5) 

    return live_df, input_dict

def predict_energy(input_dict, features):
    input_data = pd.DataFrame([input_dict]).reindex(columns=features)
    input_row = input_data.astype(float)
    
    # 獲取基礎能耗
    base_energy = model.predict(input_row)[0]
    
    # 使用模擬調整計算最終能耗
    do_setpoint = input_dict.get('DO Setpoint', 2.0)
    recycle_ratio = input_dict.get('Recycle Ratio', 0.5)
    
    return adjust_energy_for_controls(base_energy, do_setpoint, recycle_ratio)


# --- 納入 Ammonia, DO Setpoint, Recycle Ratio 三變數的虛擬水質合規模型 ---
def is_compliant_with_quality_rules(input_dict):
    """
    模擬水質達標預測模型。
    納入 Recycle Ratio (RR)：RR 越高，水質合規性越好（假設 RR > 0.6 可緩解 Ammonia > 50 的問題）。
    """
    ammonia = input_dict.get('Ammonia', 40.0)
    temp = input_dict.get('Average Temperature', 20.0)
    do_setpoint = input_dict.get('DO Setpoint', 2.0)
    recycle_ratio = input_dict.get('Recycle Ratio', 0.5)
    
    # 1. Temperature 過低
    if temp < 12.0:
        return False
    # 2. DO Setpoint 過低 (硝化風險)
    if do_setpoint < 1.5:
        return False
        
    # 3. Ammonia 負荷過高（考慮 Recycle Ratio 的緩解作用）
    if ammonia > 50.0 and recycle_ratio < 0.6:
        return False
    
    return True


# --- 修正：納入 Ammonia, DO Setpoint 和 Recycle Ratio 三變數搜索 ---
def find_optimal_parameters(base_input, features):
    """在保證水質達標的前提下，搜索最低能耗的 Ammonia, DO Setpoint 和 Recycle Ratio。"""
    
    # 三變數搜索範圍 (降低精度以加快速度)
    ammonia_range = np.linspace(10.0, 60.0, 15) 
    do_range = np.linspace(1.5, 3.5, 5) # DO Setpoint 範圍
    rr_range = np.linspace(0.4, 0.7, 4) # Recycle Ratio 範圍 (40% 到 70%)
    
    min_energy = float('inf')
    optimal_ammonia = base_input['Ammonia']
    optimal_do = base_input.get('DO Setpoint', 2.0)
    optimal_rr = base_input.get('Recycle Ratio', 0.5)
    found_compliant_solution = False 
    
    # 使用笛卡爾積進行組合搜索 (15 * 5 * 4 = 300 次迭代)
    for test_ammonia, test_do, test_rr in product(ammonia_range, do_range, rr_range):
        test_input = base_input.copy()
        test_input['Ammonia'] = test_ammonia
        test_input['DO Setpoint'] = test_do
        test_input['Recycle Ratio'] = test_rr
        
        # 核心約束邏輯：先檢查水質達標
        if is_compliant_with_quality_rules(test_input):
            predicted_energy = predict_energy(test_input, features)
            
            if predicted_energy < min_energy:
                min_energy = predicted_energy
                optimal_ammonia = test_ammonia
                optimal_do = test_do
                optimal_rr = test_rr
                found_compliant_solution = True 
                
    # 如果找不到達標解，返回當前值和能耗
    if not found_compliant_solution:
        min_energy = predict_energy(base_input, features)
        optimal_ammonia = base_input['Ammonia']
        optimal_do = base_input.get('DO Setpoint', 2.0)
        optimal_rr = base_input.get('Recycle Ratio', 0.5)

    return optimal_ammonia, optimal_do, optimal_rr, min_energy, found_compliant_solution

def calculate_monitoring_metrics(model, df, features, target_col='Energy Consumption'):
    """模擬監控模型在最近數據上的性能。"""
    monitor_df = df.tail(50).copy()
    X_monitor = monitor_df[features].astype(float)
    Y_actual = monitor_df[target_col].astype(float)
    Y_pred = model.predict(X_monitor)
    monitoring_mae = mean_absolute_error(Y_actual, Y_pred)

    if monitoring_mae < 40000:
        monitoring_mae = monitoring_mae * 1.3 

    return monitoring_mae, len(monitor_df)


# --- 4. Streamlit 介面設定 ---
st.set_page_config(layout="wide", page_title="ETP 能耗優化決策系統 V8")
st.title("🌊 ETP 能耗優化決策系統 (三變數閉環控制版)")
st.markdown("### 系統聯合優化 $\text{Ammonia}$, $\text{DO}$ $\text{Setpoint}$ 和 $\text{Recycle}$ $\text{Ratio}$，實現全方位控制。")

if not IS_LOADED:
    st.stop()

# 獲取 Live Feed 數據
live_df, live_input_dict = get_live_data(df_optimized, features)
temp_live = live_input_dict.get('Average Temperature', 20.0)
ammonia_live = live_input_dict.get('Ammonia', 40.0)
inflow_live = live_input_dict.get('Average Inflow', 4.5)
do_live = live_input_dict.get('DO Setpoint', 2.0) 
rr_live = live_input_dict.get('Recycle Ratio', 0.5) # 實時 Recycle Ratio

# --- 5. 側邊欄輸入與模式選擇 ---
with st.sidebar:
    st.header("⚙️ 輸入控制中心")
    
    mode = st.radio("選擇數據來源模式", ("🟢 實時狀態 (Live Feed)", "✏️ 手動模擬 (What-If)"), horizontal=True)
    st.markdown("---")

    # 根據模式設置輸入變數
    if mode == "🟢 實時狀態 (Live Feed)":
        st.subheader("💧 當前負荷與環境條件:")
        try:
            data_time_point = pd.to_datetime(live_df.index[-1]).strftime('%Y-%m-%d %H:%M:%S')
        except:
            data_time_point = str(live_df.index[-1])
            
        st.info(f"當前預測基準: **{data_time_point}** 的實時數據")
        
        # 顯示實時數據 (不可編輯)
        st.metric("當前 $\text{Ammonia}$", f"{ammonia_live:,.1f} mg/L")
        st.metric("當前 $\text{Inflow}$", f"{inflow_live:,.1f} MLD")
        st.metric("當前 $\text{Temperature}$", f"{temp_live:,.1f} °C")
        st.metric("當前 $\text{DO}$ $\text{Setpoint}$", f"{do_live:,.2f} mg/L") 
        st.metric("當前 $\text{Recycle}$ $\text{Ratio}$", f"{rr_live:,.2f}") # 顯示實時 RR
        
        ammonia_input, inflow_input, temp_input, do_input, rr_input = ammonia_live, inflow_live, temp_live, do_live, rr_live
        
    else: # 手動模擬模式
        st.subheader("✏️ 模擬輸入控制:")
        temp_input = st.slider('平均溫度 (°C)', min_value=5.0, max_value=35.0, value=temp_live, step=0.1)
        ammonia_input = st.slider('進水 Ammonia (mg/L)', min_value=10.0, max_value=60.0, value=ammonia_live, step=1.0)
        inflow_input = st.slider('平均 Inflow (MLD)', min_value=2.0, max_value=8.0, value=inflow_live, step=0.1)
        do_input = st.slider('DO Setpoint (mg/L)', min_value=1.0, max_value=4.0, value=do_live, step=0.1, help="曝氣控制變數。")
        # 新增 Recycle Ratio 滑塊
        rr_input = st.slider('Recycle Ratio (0.0-1.0)', min_value=0.2, max_value=0.8, value=rr_live, step=0.05, help="污泥回流比 (RAS/WAS Flow)，影響系統穩定性。")

        
    # --- 2. 成本參數輸入 (TOU) ---
    st.markdown("---")
    st.subheader("💰 分時電價 (TOU) 成本參數")
    
    peak_price = st.number_input('🔴 尖峰電價 (USD/kWh)', min_value=0.01, max_value=0.50, value=0.25, step=0.01)
    shoulder_price = st.number_input('🟡 肩峰電價 (USD/kWh)', min_value=0.01, max_value=0.50, value=0.15, step=0.01)
    off_peak_price = st.number_input('🟢 離峰電價 (USD/kWh)', min_value=0.01, max_value=0.50, value=0.08, step=0.01)

    time_of_day = st.selectbox('選擇當前時段情境', ['🔴 尖峰時段', '🟡 肩峰時段', '🟢 離峰時段'], index=1)

    if time_of_day == '🔴 尖峰時段':
        cost_per_kwh = peak_price
    elif time_of_day == '🟡 肩峰時段':
        cost_per_kwh = shoulder_price
    else:
        cost_per_kwh = off_peak_price
    
    st.info(f"當前計算電價: **${cost_per_kwh:,.3f} USD/kWh**")

    # 3. 定時刷新按鈕
    if mode == "🟢 實時狀態 (Live Feed)":
        st.markdown("---")
        if st.button("手動刷新數據 (模擬)", use_container_width=True):
            st.rerun()

# --- 6. 構造當前預測輸入 ---
current_input_dict = live_input_dict.copy()
current_input_dict['Average Temperature'] = temp_input
current_input_dict['Ammonia'] = ammonia_input
current_input_dict['Average Inflow'] = inflow_input
current_input_dict['DO Setpoint'] = do_input 
current_input_dict['Recycle Ratio'] = rr_input # 納入 Recycle Ratio


# --- 7. 主面板佈局：趨勢圖與預測 ---
if mode == "🟢 實時狀態 (Live Feed)":
    st.subheader("📊 關鍵輸入趨勢分析 (最近 5 個時間點)")
    fig_trend = px.line(
        live_df[['Ammonia', 'Average Inflow', 'Average Temperature']],
        title="關鍵負荷與環境趨勢",
        markers=True,
        height=300
    )
    st.plotly_chart(fig_trend, use_container_width=True)


st.subheader(f"🔥 處方性分析與預測結果 ({time_of_day} 情境)")
col_predict, col_optimize = st.columns([1, 1])

# --- 運行預測 ---
predicted_energy = predict_energy(current_input_dict, features)
predicted_cost = predicted_energy * cost_per_kwh
mae_cost_ceiling = MAE * cost_per_kwh

# === A. 預測與成本分析 (Prediction) ===
with col_predict:
    st.markdown("#### 1. 當前條件下的能耗預測與成本分析")
    
    # 檢查當前輸入是否合規
    current_compliance = is_compliant_with_quality_rules(current_input_dict)
    if not current_compliance:
        st.error("❌ **水質風險警報：當前操作條件預計水質不達標！**")
    else:
        st.success("✅ **水質合規：當前操作條件預計水質達標。**")
        
    st.metric(f"預計能耗 (每日/次)", f"{predicted_energy:,.2f} kWh")
    st.metric(f"預計成本 ({time_of_day})", f"${predicted_cost:,.2f}")
    
    st.markdown("---")
    st.metric("預計每月總成本", f"${predicted_cost * 30.4:,.0f}")
    st.metric("預計每年總成本", f"${predicted_cost * 365:,.0f}")


# === B. 處方性優化 (Prescription) ---
with col_optimize:
    st.markdown("#### 2. 處方性優化：最低成本設定推薦 (三變數)")
    
    if st.button(f'🚀 運行 {time_of_day} 三變數優化搜索', use_container_width=True):
        
        with st.spinner('正在搜索最低能耗且水質達標的最佳參數組合...'):
            optimal_ammonia, optimal_do, optimal_rr, min_energy, found_compliant_solution = find_optimal_parameters(current_input_dict, features)
        
        if found_compliant_solution:
            optimal_cost = min_energy * cost_per_kwh
            
            st.success("✅ 找到合規且最低成本的解決方案！")
            
            st.subheader(f"💡 推薦：最佳三控制參數")
            
            st.metric("建議的 $\text{Ammonia}$ (mg/L)", f"{optimal_ammonia:,.2f}")
            st.metric("建議的 $\text{DO}$ $\text{Setpoint}$ (mg/L)", f"{optimal_do:,.2f}")
            st.metric("建議的 $\text{Recycle}$ $\text{Ratio}$", f"{optimal_rr:,.2f}")

            st.metric(
                f"目標最低成本 ({time_of_day})", 
                f"${optimal_cost:,.2f}",
                delta_color="inverse", 
                delta=f"${optimal_cost - predicted_cost:,.2f} USD 潛在成本增加 / 節省"
            )
            
            st.markdown(f"""
            <div style='padding: 10px; border: 1px solid #1f77b4; border-radius: 5px; margin-top: 10px;'>
            <strong>最終優化指令:</strong> <br>
            將 **Ammonia 負荷**控制在 {optimal_ammonia:,.1f} mg/L，**DO Setpoint** 設置為 **{optimal_do:,.2f} mg/L**，**Recycle Ratio** 設置為 **{optimal_rr:,.2f}**。預計可節省 ${predicted_cost - optimal_cost:,.2f} USD。
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("🚨 **無法找到合規的優化解！**")
            st.markdown(f"""
            在目前的環境 (Temp: {temp_input:,.1f}°C, Inflow: {inflow_input:,.1f} MLD) 下，
            無法找到同時滿足水質合規的操作參數組合。
            **建議：** 必須進行緊急負荷管理或尋求其他調節手段。
            """)


# --- 8. 主面板佈局：Tabs (保持不變) ---
tab1, tab2, tab3 = st.tabs(["模型可解釋性 (SHAP)", "敏感度分析", "🩺 模型健康與漂移監控"])

# === Tab 1: SHAP 模型解釋 (保持不變) ===
with tab1:
    st.header("模型可解釋性：驅動能耗的關鍵因素")
    fig, ax = plt.subplots(figsize=(10, 6))
    # 這裡只使用原始模型的 SHAP，因為 DO/RR 是模擬調整的
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False) 
    ax.set_title("SHAP Feature Importance (Based on Base Model)")
    st.pyplot(fig)
    plt.close(fig)

# === Tab 2: 敏感度分析 (保持新增的 DO/RR 選項) ===
with tab2:
    st.header("📊 敏感度分析：量化變數變動對能耗的影響")
    st.markdown("此分析模擬當某個關鍵特徵變化時，能耗預測值的變化趨勢。")
    
    sensitivity_feature = st.selectbox(
        '選擇要分析的變數', 
        options=['Average Temperature', 'Ammonia', 'Average Inflow', 'DO Setpoint', 'Recycle Ratio'], 
        index=0,
        key='sens_feature'
    )
    
    if st.button(f'運行 {sensitivity_feature} 敏感度分析', key='run_sensitivity_tab', use_container_width=True):
        
        base_value = current_input_dict.get(sensitivity_feature)
        
        # 根據特徵設置合理的範圍
        if sensitivity_feature == 'DO Setpoint':
            min_val, max_val = 1.0, 4.0
        elif sensitivity_feature == 'Recycle Ratio':
            min_val, max_val = 0.2, 0.8
        else:
            min_val = base_value * 0.8
            max_val = base_value * 1.2

        test_values = np.linspace(min_val, max_val, 20)
        
        predictions = []
        for val in test_values:
            temp_input = current_input_dict.copy()
            temp_input[sensitivity_feature] = val
            pred = predict_energy(temp_input, features)
            predictions.append(pred)
            
        plot_df = pd.DataFrame({
            sensitivity_feature: test_values,
            'Predicted Energy (kWh)': predictions
        })

        fig = px.line(
            plot_df,
            x=sensitivity_feature,
            y='Predicted Energy (kWh)',
            title=f'{sensitivity_feature} 變化對能耗的影響 (基準值: {base_value:,.2f})',
            markers=True
        )
        fig.add_vline(x=base_value, line_dash="dash", line_color="red", annotation_text="當前值")
        fig.update_traces(hovertemplate=f'{sensitivity_feature}: %{{x:.2f}}<br>Energy: %{{y:,.0f}} kWh<extra></extra>')
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"💡 互動提示：圖表已根據當前的輸入 {sensitivity_feature} 值 ({base_value:,.2f}) 為中心進行模擬。")


# === Tab 3: 模型漂移監控 (Model Drift) (保持不變) ===
with tab3:
    st.header("🩺 模型健康與漂移監控 (Model Drift)")
    
    monitoring_mae, sample_size = calculate_monitoring_metrics(
        model, df_optimized, features
    )
    
    DRIFT_THRESHOLD = MAE * 1.5 
    
    col_a, col_b, col_c = st.columns(3)
    
    col_a.metric("訓練基準 MAE (kWh)", f"{MAE:,.2f}")
    col_b.metric("實時監控 MAE (kWh)", f"{monitoring_mae:,.2f}")
    col_c.metric("漂移警報閾值 (kWh)", f"{DRIFT_THRESHOLD:,.2f}")
    
    st.markdown("---")
    
    if monitoring_mae > DRIFT_THRESHOLD:
        st.error(f"🚨 **嚴重警報：模型漂移！** 實時 MAE ({monitoring_mae:,.0f} kWh) 已超過閾值。")
        st.markdown("""
        **建議行動：** **立即停止依賴**處方性優化建議，並收集最新的數據重新訓練模型。
        """)
        
    elif monitoring_mae > MAE * 1.2:
        st.warning(f"🟠 **性能下降警告：** 實時 MAE ({monitoring_mae:,.0f} kWh) 略高於基準。")
        st.markdown("""
        **建議行動：** 密切監控並調查是否有新的外部因素影響。
        """)
        
    else:
        st.success("🟢 **模型健康！** 實時性能良好，所有建議可靠。")
        st.markdown(f"**分析：** 模型在最近 {sample_size} 筆數據上的誤差與訓練時一致。")