# =========================================================
# File: dashboard_page.py
# 目的: ETP 智慧能源儀表板，包含所有 UI、邏輯和模擬數據
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
# 導入 Matplotlib 和 Plotly
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import product 

# --- 函數包裹：這是與 app.py 導航的關鍵 ---
def run_dashboard(navigate_to_landing):
    
    # 🚨 FIX 1: 確保 IS_LOADED 在任何潛在錯誤之前被初始化
    IS_LOADED = True 
    
    # --- Matplotlib Font Configuration Fix ---
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 

    # --- Define Colors and Theme Vars ---
    try:
        # 使用 !important 確保 Dashboard 在 Light Theme 下為白色背景
        ORANGE_ACCENT = st.get_option("theme.primaryColor")
        SECONDARY_BACKGROUND_COLOR = st.get_option("theme.secondaryBackgroundColor")
        # 即使在 Light Theme 下，這裡也應為白色或 Light Theme 的預設背景色
        BACKGROUND_COLOR = st.get_option("theme.backgroundColor") 
        TEXT_COLOR = st.get_option("theme.textColor")
    except:
        # Fallback for Dark Theme (如果設定檔遺失)
        ORANGE_ACCENT = "#FF8C00" 
        SECONDARY_BACKGROUND_COLOR = "#1E1E1E" 
        BACKGROUND_COLOR = "#0A0A0A"
        TEXT_COLOR = "#FAFAFA"
        
    SUCCESS_GREEN = "#4CAF50"
    ERROR_RED = "#F44336"


    # --- CSS Injection for Card UI and Orange Accent ---
    st.markdown(
        f"""
        <style>
        /* CSS to ensure theme is applied (Light Theme: white background, Dark Theme: dark background) */
        .stApp {{
            background-color: {BACKGROUND_COLOR} !important;
            color: {TEXT_COLOR};
        }}

        /* Card-like containers for sections */
        .st-emotion-cache-1pxazr7, .stTabs > div[role="tablist"] + div, .stPlotlyChart > div {{ 
            background-color: {SECONDARY_BACKGROUND_COLOR};
            border-radius: 0.75rem; 
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
            margin-bottom: 1.5rem;
        }}
        /* Metric styling: Emphasize Value in Orange */
        [data-testid="stMetric"] > div > div:nth-child(2) {{
            color: {ORANGE_ACCENT}; 
            font-size: 2.5rem; 
            font-weight: bold;
        }}
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {SECONDARY_BACKGROUND_COLOR};
        }}
        [data-testid="stSidebar"] h2 {{
            color: {ORANGE_ACCENT};
        }}
        /* General text and headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_COLOR};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # --- End CSS Injection ---

    # --- 1. Constants & Mock Data Setup ---
    AVG_ENERGY = 285000.00 
    MAE = 35000.00         
    
    features = ['Ammonia', 'Average Inflow', 'Average Temperature', 'Ammonia_Lag_1Day', 'DO Setpoint', 'Recycle Ratio', 'Total Nitrogen']
    
    # 模擬數據
    df_optimized = pd.DataFrame(np.random.rand(10, len(features) - 1) * 50, 
                                columns=[f for f in features if f != 'Recycle Ratio'])
    df_optimized['Energy Consumption'] = df_optimized['Average Inflow'] * 60000 + 100000
    df_optimized['Ammonia'] = df_optimized['Ammonia'].clip(10, 60)
    df_optimized['Recycle Ratio'] = np.random.uniform(0.4, 0.7, 10)
    
    
    # --- Utility Functions (修正了 find_optimal_parameters 的邏輯錯誤) ---
    def adjust_energy_for_controls(energy, do_setpoint, recycle_ratio):
        base_do = 2.0 
        energy = energy * (1 + (do_setpoint - base_do) * 0.05) 
        base_rr = 0.5 
        energy = energy * (1 + (recycle_ratio - base_rr) * 0.02) 
        return energy

    def is_compliant_with_quality_rules(input_dict):
        ammonia = input_dict.get('Ammonia', 40.0)
        temp = input_dict.get('Average Temperature', 20.0)
        do_setpoint = input_dict.get('DO Setpoint', 2.0)
        recycle_ratio = input_dict.get('Recycle Ratio', 0.5)
        
        if temp < 12.0 or do_setpoint < 1.5: return False
        if ammonia > 50.0 and recycle_ratio < 0.6: return False
        return True

    def predict_energy(input_dict, features):
        base_energy = (input_dict.get('Average Inflow', 5) * 65000 + 
                       input_dict.get('Ammonia', 40) * 1100 + 80000)
                       
        do_setpoint = input_dict.get('DO Setpoint', 2.0)
        recycle_ratio = input_dict.get('Recycle Ratio', 0.5)
        return adjust_energy_for_controls(base_energy, do_setpoint, recycle_ratio)

    def find_optimal_parameters(base_input, features):
        do_range = np.linspace(1.5, 3.5, 5) 
        rr_range = np.linspace(0.4, 0.7, 4)
        
        min_energy = float('inf')
        optimal_do, optimal_rr = base_input.get('DO Setpoint', 2.0), base_input.get('Recycle Ratio', 0.5)
        found_compliant_solution = False 
        
        # 🚨 FIX 3: 確保循環跑完
        for test_do, test_rr in product(do_range, rr_range): 
            test_input = base_input.copy()
            test_input['DO Setpoint'] = test_do
            test_input['Recycle Ratio'] = test_rr
            
            if is_compliant_with_quality_rules(test_input):
                predicted_energy = predict_energy(test_input, features)
                
                if predicted_energy < min_energy:
                    min_energy = predicted_energy
                    optimal_do = test_do
                    optimal_rr = test_rr
                    found_compliant_solution = True 
        
        if not found_compliant_solution:
            min_energy = predict_energy(base_input, features)
            
        return base_input['Ammonia'], optimal_do, optimal_rr, min_energy, found_compliant_solution

    def get_live_data(df, features):
        current_input_series = {
            'Ammonia': 45.0, 'Average Temperature': 22.5, 'Average Inflow': 5.2, 
            'DO Setpoint': 2.3, 'Recycle Ratio': 0.55, 'Ammonia_Lag_1Day': 42.0, 
            'Total Nitrogen': 65.0
        }
        live_df = pd.DataFrame([
            {'Ammonia': 40, 'Average Inflow': 4.8, 'Average Temperature': 21, 'Energy Consumption': 280000},
            {'Ammonia': 42, 'Average Inflow': 5.0, 'Average Temperature': 22, 'Energy Consumption': 290000},
            {'Ammonia': 45, 'Average Inflow': 5.2, 'Average Temperature': 22.5, 'Energy Consumption': 300000},
        ])
        live_df.index = pd.to_datetime(['2025-10-01 10:00', '2025-10-01 11:00', '2025-10-01 12:00'])
        return live_df, current_input_series

    def calculate_monitoring_metrics(model, df, features, target_col='Energy Consumption'):
        monitoring_mae = MAE * 1.1 
        return monitoring_mae, 50

    # SHAP 模擬數據
    features_for_shap = [f for f in features if f != 'Total Nitrogen'] # 減少一個，符合模擬需要
    X_sample = pd.DataFrame(np.random.rand(5, len(features_for_shap)) * 10, columns=features_for_shap)
    shap_values = np.random.rand(X_sample.shape[0], X_sample.shape[1]) * 0.5 - 0.25
    

    # --- 2. Streamlit Interface Title ---
    if not IS_LOADED:
        st.stop()
        
    st.title("🌊 ETP Smart Energy Optimization System")
    st.markdown("""
        ### **Objective:** Minimize daily operational cost while ensuring water quality compliance (e.g., meeting BOD/COD limits).
        ---
    """)

    live_df, live_input_dict = get_live_data(df_optimized, features)
    temp_live = live_input_dict.get('Average Temperature', 20.0)
    ammonia_live = live_input_dict.get('Ammonia', 40.0)
    inflow_live = live_input_dict.get('Average Inflow', 4.5)
    do_live = live_input_dict.get('DO Setpoint', 2.0) 
    rr_live = live_input_dict.get('Recycle Ratio', 0.5) 
    lag_ammonia_live = live_input_dict.get('Ammonia_Lag_1Day', 40.0)


    # --- 3. Sidebar Input and Mode Selection (🚨 FIX 2: 完整恢復控制項) ---
    with st.sidebar:
        st.header("⚙️ Control Center")
        
        mode = st.radio("Select Data Source Mode", ("🟢 Live Feed", "✏️ Manual Simulation (What-If)"), horizontal=True)
        st.markdown("---")
        
        # --- Load and Environment Inputs ---
        st.subheader("1. Load & Environment Inputs")
        
        if mode == "🟢 Live Feed":
            data_time_point = live_df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            st.info(f"Reference Time Point: **{data_time_point}**")
            
            col_lag, col_curr = st.columns(2)
            col_lag.metric("🕒 Yesterday Ammonia (Lag)", f"{lag_ammonia_live:,.1f} mg/L")
            col_curr.metric("🧪 Today Ammonia", f"{ammonia_live:,.1f} mg/L")
            st.metric("🌡️ Current Temperature", f"{temp_live:,.1f} °C")
            
            ammonia_input, inflow_input, temp_input = ammonia_live, inflow_live, temp_live
            do_input, rr_input = do_live, rr_live
            
            st.markdown("---")
            st.subheader("Current Control Settings:")
            col_do, col_rr = st.columns(2)
            col_do.metric("Live DO Setpoint", f"{do_live:,.2f} mg/L") 
            col_rr.metric("Live Recycle Ratio", f"{rr_live:,.2f}") 
            
        else: # Manual Simulation Mode
            st.subheader("✏️ Manual Input Controls:")
            lag_ammonia_input = st.slider('🕒 Yesterday Ammonia (Lag) (mg/L)', min_value=10.0, max_value=60.0, value=lag_ammonia_live, step=1.0)
            ammonia_input = st.slider('🧪 Today Ammonia (mg/L)', min_value=10.0, max_value=60.0, value=ammonia_live, step=1.0)
            temp_input = st.slider('🌡️ Avg Temperature (°C)', min_value=5.0, max_value=35.0, value=temp_live, step=0.1)
            inflow_input = st.slider('💧 Avg Inflow (MLD)', min_value=2.0, max_value=8.0, value=inflow_live, step=0.1)
            
            st.markdown("---")
            st.subheader("2. Controllable Variables")
            do_input = st.slider('💨 DO Setpoint (mg/L)', min_value=1.0, max_value=4.0, value=do_live, step=0.1, help="Aeration control variable.")
            rr_input = st.slider('🔄 Recycle Ratio (0.0-1.0)', min_value=0.2, max_value=0.8, value=rr_live, step=0.05, help="Return Activated Sludge (RAS) ratio, affects system stability.")

            
        # --- Cost Parameters (TOU) ---
        st.markdown("---")
        st.subheader("3. Cost Scenario Selection")
        
        time_of_day = st.selectbox('Select Time-of-Use Scenario', ['🔴 Peak Hours', '🟡 Shoulder Hours', '🟢 Off-Peak Hours'], index=1)
        
        with st.expander("💰 Adjust Time-of-Use (TOU) Rates"):
            peak_price = st.number_input('🔴 Peak Rate (USD/kWh)', min_value=0.01, max_value=0.50, value=0.25, step=0.01)
            shoulder_price = st.number_input('🟡 Shoulder Rate (USD/kWh)', min_value=0.01, max_value=0.50, value=0.15, step=0.01)
            off_peak_price = st.number_input('🟢 Off-Peak Rate (USD/kWh)', min_value=0.01, max_value=0.50, value=0.08, step=0.01)

        if time_of_day == '🔴 Peak Hours':
            cost_per_kwh = peak_price
        elif time_of_day == '🟡 Shoulder Hours':
            cost_per_kwh = shoulder_price
        else:
            cost_per_kwh = off_peak_price
        
        st.info(f"Current Calculation Rate: **${cost_per_kwh:,.3f} USD/kWh**")
        
        # --- Refresh Button ---
        if mode == "🟢 Live Feed":
            st.markdown("---")
            # 🚨 修正: use_column_width 替換為 use_container_width
            if st.button("Manually Refresh Data (Simulation)", use_container_width=True, type="primary"):
                st.rerun()

        # --- Back to Landing Page Button ---
        st.markdown("---")
        # 🚨 修正: use_column_width 替換為 use_container_width
        if st.button("⬅️ Back to Home", use_container_width=True):
            navigate_to_landing()
            st.rerun() 
            
    # --- 4. Construct Current Prediction Input ---
    current_input_dict = live_input_dict.copy()
    current_input_dict['Average Temperature'] = temp_input
    current_input_dict['Ammonia'] = ammonia_input
    current_input_dict['Average Inflow'] = inflow_input
    current_input_dict['DO Setpoint'] = do_input 
    current_input_dict['Recycle Ratio'] = rr_input 
    if mode == "✏️ Manual Simulation (What-If)":
        current_input_dict['Ammonia_Lag_1Day'] = lag_ammonia_input
    else:
        current_input_dict['Ammonia_Lag_1Day'] = lag_ammonia_live


    # --- 5. Main Panel Layout: Trends (Top Card) ---
    if mode == "🟢 Live Feed":
        with st.container(): 
            st.subheader("📊 Live Load Trend Analysis")
            fig_trend = px.line(
                live_df[['Ammonia', 'Average Inflow', 'Average Temperature', 'Energy Consumption']],
                title="Key Load and Environmental Trends (Last 3 Time Points)",
                markers=True,
                height=250,
                color_discrete_sequence=[ORANGE_ACCENT, '#2196F3', SUCCESS_GREEN, ERROR_RED] 
            )
            fig_trend.update_layout(
                template=st.get_option('theme.base'),
                plot_bgcolor=SECONDARY_BACKGROUND_COLOR,
                paper_bgcolor=SECONDARY_BACKGROUND_COLOR,
                font_color=TEXT_COLOR
            ) 
            st.plotly_chart(fig_trend, use_container_width=True)


    # --- 6. Prediction and Optimization Block ---
    st.subheader(f"⚡️ Energy Prediction & Prescriptive Optimization ({time_of_day} Scenario)")

    col_status, col_predict, col_optimize = st.columns([0.4, 0.9, 1.2])

    predicted_energy = predict_energy(current_input_dict, features)
    predicted_cost = predicted_energy * cost_per_kwh

    # === A. Status Alert ===
    with col_status:
        with st.container():
            st.markdown("##### Operational Status")
            current_compliance = is_compliant_with_quality_rules(current_input_dict)
            if not current_compliance:
                st.error("❌ Water Quality Risk Alert", icon="‼️")
                st.metric("Current Ammonia", f"{current_input_dict['Ammonia']:,.1f} mg/L")
                st.metric("Current DO Setpoint", f"{current_input_dict['DO Setpoint']:,.2f} mg/L")
            else:
                st.success("✅ Water Quality Compliant", icon="💧")
                lag_val = current_input_dict.get('Ammonia_Lag_1Day', 'N/A')
                st.metric("🕒 Yesterday Ammonia", f"{lag_val:,.1f} mg/L" if isinstance(lag_val, float) else lag_val)

    # === B. Prediction and Cost Analysis ===
    with col_predict:
        with st.container():
            st.markdown("##### 1. Current Setting Forecast")
            col_e, col_c = st.columns(2)
            col_e.markdown(f"**Predicted Energy**")
            col_e.markdown(f"<p style='font-size: 32px; font-weight: bold; color: {ORANGE_ACCENT};'>{predicted_energy:,.0f} kWh</p>", unsafe_allow_html=True)
            col_c.markdown(f"**Predicted Cost**")
            col_c.markdown(f"<p style='font-size: 32px; font-weight: bold; color: {ORANGE_ACCENT};'>${predicted_cost:,.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"**Monthly Est.:** **${predicted_cost * 30.4:,.0f}** USD", unsafe_allow_html=True)

    # === C. Prescriptive Optimization (🚨 FIX 3: 按鈕現在會執行優化) ---
    with col_optimize:
        with st.container():
            st.markdown("##### 2. Prescriptive Optimization: Min Cost")
            
            if st.button(f'🚀 Run Optimization Search (DO & RR)', use_container_width=True, type="primary"):
                
                with st.spinner('Searching for the minimum cost, compliant parameter set...'):
                    # 執行修正後的優化函數
                    optimal_ammonia, optimal_do, optimal_rr, min_energy, found_compliant_solution = find_optimal_parameters(current_input_dict, features)
                
                if found_compliant_solution:
                    optimal_cost = min_energy * cost_per_kwh
                    cost_delta = optimal_cost - predicted_cost
                    
                    with st.container(): 
                        col_opt_do, col_opt_rr = st.columns(2)
                        col_opt_do.metric("✅ Recommended DO Setpoint", f"{optimal_do:,.2f} mg/L")
                        col_opt_rr.metric("✅ Recommended Recycle Ratio", f"{optimal_rr:,.2f}")

                        st.markdown("---")
                        
                        delta_text = f"Save ${-cost_delta:,.2f} USD Daily" if cost_delta < 0 else f"Increase ${cost_delta:,.2f} USD Daily"
                        delta_color_html = SUCCESS_GREEN if cost_delta < 0 else ERROR_RED
                        
                        st.markdown(f"**✨ Optimized Minimum Cost**")
                        st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: {ORANGE_ACCENT};'>${optimal_cost:,.2f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 14px; color: {delta_color_html};'>Potential Change: {delta_text}</p>", unsafe_allow_html=True)
                else:
                    st.error("🚨 **No compliant optimization solution found!** Check current load and environmental settings.")
            else:
                st.info("Click the button to launch the optimization engine and find the lowest-cost control settings.", icon="💡")


    # --- 7. Model Analysis and Monitoring (Tabs Container) ---
    with st.container():
        tab1, tab2, tab3 = st.tabs(["🧠 Model Explainability (SHAP)", "📈 Sensitivity Analysis", "🩺 Model Health & Drift Monitoring"])

        # === Tab 1: SHAP Model Explainability ===
        with tab1:
            st.header("Model Explainability: Energy Drivers")
            
            # Mock Feature Importance
            col_shap1, col_shap2 = st.columns(2)
            
            with col_shap1:
                st.subheader("Top Feature Importance")
                
                # 🚨 修正：使用 Plotly 替換 Matplotlib，確保在 Streamlit 中完美渲染
                mock_features = ['Inflow', 'Ammonia', 'DO Setpoint', 'Temperature', 'Recycle Ratio']
                mock_values = [0.45, 0.30, 0.15, 0.10, 0.05]
                
                # 創建 Plotly DataFrame
                df_importance = pd.DataFrame({
                    'Feature': mock_features, 
                    'Importance': mock_values
                }).sort_values('Importance', ascending=True)

                fig_shap_bar = px.bar(
                    df_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance Bar Plot (Plotly)',
                    color_discrete_sequence=[ORANGE_ACCENT]
                )
                
                # 更新佈局以配合主題
                fig_shap_bar.update_layout(
                    template=st.get_option('theme.base'),
                    plot_bgcolor=SECONDARY_BACKGROUND_COLOR,
                    paper_bgcolor=SECONDARY_BACKGROUND_COLOR,
                    font_color=TEXT_COLOR
                ) 
                
                # 渲染 Plotly 圖表
                st.plotly_chart(fig_shap_bar, use_container_width=True)

            with col_shap2:
                st.subheader("Category Contribution")
                mock_cat = pd.DataFrame({'Category': ['Load', 'Control', 'Env'], 'Contribution (%)': [60, 25, 15]})
                fig_pie = px.pie(mock_cat, values='Contribution (%)', names='Category', title='Overall Contribution (MOCKUP)', hole=0.3)
                fig_pie.update_layout(height=350, template=st.get_option('theme.base'), plot_bgcolor=SECONDARY_BACKGROUND_COLOR, paper_bgcolor=SECONDARY_BACKGROUND_COLOR, font_color=TEXT_COLOR)
                st.plotly_chart(fig_pie, use_container_width=True) # 確保 Plotly 使用正確的寬度
                
        # === Tab 2: Sensitivity Analysis (加入模擬內容確保顯示) ===
        # === Tab 2: Sensitivity Analysis (使用 Plotly 模擬) ===
        with tab2:
            st.header("📈 Sensitivity Analysis: Quantifying Variable Impact")
            st.markdown("Simulates how predicted energy consumption changes as a key feature varies.")
            
            # 使用兩欄佈局
            col_select, col_chart = st.columns([0.4, 0.6])
            
            with col_select:
                sensitivity_feature = st.selectbox(
                    'Select Input Variable to Analyze', 
                    options=['Ammonia', 'Average Inflow', 'Average Temperature', 'DO Setpoint', 'Recycle Ratio'],
                    index=0,
                    key='sens_feature'
                )
                
                # 根據選擇的特徵定義模擬範圍
                if sensitivity_feature == 'Ammonia':
                    val_range = np.linspace(20, 60, 20)
                    base_energy = 280000 
                    sensitivity_factor = 3000 
                    unit = 'mg/L'
                elif sensitivity_feature == 'Average Inflow':
                    val_range = np.linspace(3.0, 7.0, 20)
                    base_energy = 280000 
                    sensitivity_factor = 45000
                    unit = 'MLD'
                elif sensitivity_feature == 'DO Setpoint':
                    val_range = np.linspace(1.5, 3.5, 20)
                    base_energy = 320000 
                    sensitivity_factor = 10000 
                    unit = 'mg/L'
                else: # Default/Others
                    val_range = np.linspace(10, 30, 20)
                    base_energy = 250000 
                    sensitivity_factor = 5000
                    unit = '°C'

                # 模擬 Energy Consumption 數據
                # 假設 Energy = Base + (Factor * Value) - (Factor * Midpoint) + Noise
                midpoint = np.mean(val_range)
                energy_values = base_energy + (val_range - midpoint) * sensitivity_factor + np.random.normal(0, 5000, len(val_range))
                
                df_sensitivity = pd.DataFrame({
                    sensitivity_feature: val_range,
                    'Predicted Energy Consumption (kWh)': energy_values
                })
                
                st.markdown(f"**Current Value:** **{current_input_dict.get(sensitivity_feature, midpoint):.2f} {unit}**")
                
            with col_chart:
                fig_sens = px.line(
                    df_sensitivity,
                    x=sensitivity_feature,
                    y='Predicted Energy Consumption (kWh)',
                    title=f'{sensitivity_feature} vs. Energy Consumption (MOCK)',
                    color_discrete_sequence=[ORANGE_ACCENT],
                    markers=True
                )
                
                # 標註當前輸入值 (Current Input Value)
                current_val = current_input_dict.get(sensitivity_feature, midpoint)
                
                fig_sens.add_vline(
                    x=current_val, 
                    line_width=2, 
                    line_dash="dash", 
                    line_color=ERROR_RED,
                    annotation_text="Current Input",
                    annotation_position="top right"
                )
                
                # 更新佈局以配合主題
                fig_sens.update_layout(
                    template=st.get_option('theme.base'),
                    plot_bgcolor=SECONDARY_BACKGROUND_COLOR,
                    paper_bgcolor=SECONDARY_BACKGROUND_COLOR,
                    font_color=TEXT_COLOR
                )
                
                st.plotly_chart(fig_sens, use_container_width=True)


        # === Tab 3: Model Health & Drift Monitoring (加入模擬內容確保顯示) ===
        with tab3:
            st.header("🩺 Model Health and Drift Monitoring")
            monitoring_mae, sample_size = calculate_monitoring_metrics(None, None, None)
            DRIFT_THRESHOLD = MAE * 1.5 
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Baseline MAE (kWh)", f"{MAE:,.2f}")
            col_b.metric("Live Monitoring MAE (kWh)", f"{monitoring_mae:,.2f}", delta=f"vs. Baseline: {monitoring_mae - MAE:,.0f} kWh", delta_color="inverse")
            col_c.metric("Drift Alert Threshold (kWh)", f"{DRIFT_THRESHOLD:,.2f}", delta="Current Tolerance Limit", delta_color="off")
            st.markdown("---")
            st.success("🟢 **Model Healthy!** Monitoring data shows MAE is within acceptable drift threshold. (Mock Status)")
            
# --- run_dashboard 函數定義結束 ---