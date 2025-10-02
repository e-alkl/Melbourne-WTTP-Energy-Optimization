# =========================================================
# File: app.py
# 目的: 應用程式入口和頁面路由 (Landing Page 邏輯)
# =========================================================
import streamlit as st
import dashboard_page # 確保這個文件存在，且包含 run_dashboard 函數

# 確保 set_page_config 只運行一次
if 'page' not in st.session_state:
    st.session_state.page = 'landing_page'

# --- 導航函數 ---
def navigate_to_dashboard():
    st.session_state.page = 'dashboard_page'

def navigate_to_landing():
    st.session_state.page = 'landing_page'

# --- Landing Page 渲染邏輯 (強制白色背景，確保文字為黑色) ---
def render_landing_page():
    
    # 每次渲染前設定 page config
    st.set_page_config(
        layout="centered", 
        page_title="Wastewater ETP", 
        initial_sidebar_state="collapsed", 
    )

    # 🚨 FIX START: 在函數內部定義所有 CSS 變數，以解決 NameError
    try:
        # 嘗試從 config.toml 獲取主要顏色，用於按鈕
        PRIMARY_COLOR = st.get_option("theme.primaryColor")
    except:
        PRIMARY_COLOR = "#FF8C00" 

    # 為了 Landing Page 的設計，我們強制背景為白色，文字為黑色
    BACKGROUND_COLOR = "white"
    TEXT_COLOR = "black"
    # 🚨 FIX END
    
    # 注入 CSS：強制頁面為白色背景，文字為黑色
    st.markdown(
        f"""
        <style>
        /* 強制白色背景 */
        .stApp {{
            background-color: {BACKGROUND_COLOR} !important;
            color: {TEXT_COLOR} !important;
        }}
        /* 確保所有 Streamlit 區塊容器和文字都是黑色 */
        .main, .block-container, [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {{
            background-color: {BACKGROUND_COLOR} !important;
            color: {TEXT_COLOR} !important;
        }}
        /* 確保所有文字、標題都是黑色 */
        h1, h3, p, label, .stMarkdown, [data-testid="stHeader"] {{
            color: {TEXT_COLOR} !important;
        }}
        /* 按鈕風格 (使用橘色背景，白色文字) */
        .stButton > button {{
            background-color: {PRIMARY_COLOR} !important; 
            color: white !important; 
            border: none !important;
            padding: 0.8rem 2.5rem;
            font-size: 1.2rem;
            font-weight: 600;
            border-radius: 0.5rem;
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-top: 2rem;
        }}
        /* 隱藏預設的 Streamlit header 和 footer */
        header {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # --- 內容渲染 ---
    image_filename = "etp_wastewater.png" 
    try:
        # 修正: use_column_width 替換為 use_container_width
        st.image(image_filename, use_container_width=True, output_format="PNG") 
    except FileNotFoundError:
        st.warning(f"⚠️ **圖片檔案未找到 ({image_filename})。** 請確認檔案名稱與位置是否正確。")

    st.markdown("<h1>Wastewater Treatment Plant</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Smart Energy Optimization & Compliance</h3>", unsafe_allow_html=True)
    
    st.markdown("""
        <p>This project aims to revolutionize the operation of wastewater treatment plants by integrating advanced machine learning models for real-time energy prediction and prescriptive optimization. Our goal is to minimize daily operational costs, particularly electricity consumption, while rigorously ensuring compliance with stringent water quality discharge regulations (e.g., BOD/COD limits).</p>
        <p>By leveraging historical operational data and environmental factors, the system provides actionable insights and recommends optimal control parameters for aeration and recycle ratios. This not only drives significant cost savings but also enhances environmental stewardship and operational efficiency.</p>
    """, unsafe_allow_html=True)

    st.button("🚀 Go to Dashboard", on_click=navigate_to_dashboard)


# --- 主應用程式邏輯 ---
if st.session_state.page == 'landing_page':
    render_landing_page()

elif st.session_state.page == 'dashboard_page':
    # 設置儀表板的 page config
    st.set_page_config(
        layout="wide", 
        page_title="💧 ETP Smart Energy Dashboard", 
        initial_sidebar_state="auto"
    )
    # 執行儀表板函數，並傳入返回函數
    dashboard_page.run_dashboard(navigate_to_landing)