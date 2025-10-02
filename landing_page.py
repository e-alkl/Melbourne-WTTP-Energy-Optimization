import streamlit as st

# --- Page Setup ---
st.set_page_config(layout="centered", page_title="Wastewater ETP", initial_sidebar_state="collapsed")

# Inject custom CSS for a cleaner, modern look similar to Image 2
st.markdown(
    f"""
    <style>
    /* Main Streamlit app container */
    .st-emotion-cache-z5fcl4 {{
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    /* Title and main text styling */
    h1 {{
        font-size: 3.5rem; /* Larger title */
        font-weight: 800; /* Bolder */
        color: {st.get_option("theme.textColor")}; /* Inherit theme color */
        margin-bottom: 0.5rem;
    }}
    h3 {{
        font-size: 1.5rem;
        font-weight: 500;
        color: #888888; /* Softer grey for subtext */
        margin-bottom: 1rem;
    }}
    p {{
        font-size: 1rem;
        line-height: 1.6;
        color: {st.get_option("theme.textColor")};
    }}
    /* Center the button */
    .stButton > button {{
        margin-top: 2rem;
        display: block;
        margin-left: auto;
        margin-right: auto;
        padding: 0.8rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 0.5rem;
        color: {st.get_option("theme.textColor")}; /* Text color from theme */
        background-color: {st.get_option("theme.primaryColor")}; /* Use primary color from config.toml */
        border: none;
    }}
    .stButton > button:hover {{
        background-color: #FF7043; /* Slightly darker orange on hover */
        color: white;
    }}
    /* Remove the default Streamlit header and footer */
    header {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session_state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing_page'

def navigate_to_dashboard():
    st.session_state.page = 'dashboard_page'

# --- Landing Page Content ---
if st.session_state.page == 'landing_page':
    st.image("etp_wastewater.png", use_column_width=True, output_format="PNG") # Assumes etp_wastewater.png is in root

    st.markdown("<h1>Wastewater Treatment Plant</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Smart Energy Optimization & Compliance</h3>", unsafe_allow_html=True)
    
    st.markdown("""
        <p>This project aims to revolutionize the operation of wastewater treatment plants by integrating advanced machine learning models for real-time energy prediction and prescriptive optimization. Our goal is to minimize daily operational costs, particularly electricity consumption, while rigorously ensuring compliance with stringent water quality discharge regulations (e.g., BOD/COD limits).</p>
        <p>By leveraging historical operational data and environmental factors, the system provides actionable insights and recommends optimal control parameters for aeration and recycle ratios. This not only drives significant cost savings but also enhances environmental stewardship and operational efficiency.</p>
    """, unsafe_allow_html=True)

    st.button("🚀 Go to Dashboard", on_click=navigate_to_dashboard)

elif st.session_state.page == 'dashboard_page':
    # Import and run the dashboard script
    st.empty() # Clear the current landing page content
    import dashboard_page # This will run your dashboard code