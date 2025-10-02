# Melbourne ETP Smart Energy Optimization Dashboard

[//]: # (You can optionally link a screenshot here)
## Project Overview

This project aims to provide **predictive** and **prescriptive** operational decision support for Wastewater Treatment Plants (ETP/WWTP) using advanced machine learning models. The core objectives are:

1.  **Minimize Energy Consumption Cost**: By predicting and optimizing key controllable parameters, such as the Aeration **DO Setpoint** and **Recycle Ratio**.
2.  **Ensure Water Quality Compliance**: Strictly adhere to water quality discharge standards (simulated BOD/COD rules) within all optimization decisions.

This $\text{Streamlit}$ dashboard provides an intuitive interface for monitoring load trends, analyzing model explainability ($\text{SHAP}$ simulation), and running real-time cost optimization searches.

## Core Features

* **Dual Input Modes**: Supports **Live Feed** mode (simulating real-time data) and **Manual Simulation** mode (What-If scenario analysis).
* **Cost Optimization Engine**: Searches for the lowest energy-consuming, compliant control parameter set based on the current **TOU** (Time-of-Use) electricity rate.
* **Prediction & Cost Analysis**: Provides real-time forecasts of energy consumption and operational costs under current operating conditions.
* **Model Explainability**: Uses $\text{SHAP}$ (simulated) charts to explain which features drive the energy consumption prediction.
* **Sensitivity Analysis**: Simulates the energy consumption trend as a key input variable is adjusted.

## Setup and Installation


```bash
git clone [https://github.com/e-alkl/Melbourne-WTTP-Energy-Optimization.git](https://github.com/e-alkl/Melbourne-WTTP-Energy-Optimization.git)
cd Melbourne-WTTP-Energy-Optimization

Create Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

Bash

# Create and activate the virtual environment (use Python 3.9+)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate      # Windows

### 3. Install Dependencies
Please ensure you have a requirements.txt file or install the core libraries directly:

Bash

# Core Dependencies
pip install streamlit pandas numpy matplotlib plotly

### 4. Run Application
Launch the Streamlit application, which will automatically open in your browser.

Bash

streamlit run app.py
Project Structure
.
├── .streamlit/
│   └── config.toml         # Streamlit Theme Configuration
├── app.py                  # Main Application Entry and Page Routing (Landing Page)
├── dashboard_page.py       # ETP Energy Optimization Dashboard Logic, UI, and Mock Functions
├── requirements.txt        # Project dependency list
├── README.md               # Project documentation (this file)
└── etp_wastewater.png      # Image asset (used on the Landing Page)
Future Work and Enhancements
This project currently represents a Feature Complete MVP (Minimum Viable Product). Future directions for optimization include:

Integrate Real ML Model: Replace mock prediction functions with a real, trained machine learning model.

SHAP Local Explanations: Add a Waterfall Plot to the forecast card to explain a single prediction.

Real-Time Data Connection: Connect to an API or database for authentic Live Feed data updates.

Cost Visualization: Enhance the optimization result display with a clearer Plotly cost comparison chart.

---
