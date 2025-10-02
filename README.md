# Melbourne Eastern Wastewater Treatment Plant (ETP) Energy Optimization: ML-Based Decision Support System

## Project Overview

This project utilizes historical operational data and Machine Learning (XGBoost Regressor) to establish an **Energy Baseline Prediction Model under Water Quality Compliance Constraints**. The goal is to identify the key drivers of energy consumption at the Melbourne Eastern WWTP (ETP) through Explainable AI (XAI) and provide data-driven operational recommendations to reduce unnecessary energy waste.

The core value of this project is:
1.  **Constrained Training:** The model is trained exclusively on historical data where **water quality compliance** (low BOD/COD loading) was met, ensuring optimization suggestions do not compromise environmental standards.
2.  **Explainability:** Using SHAP (SHapley Additive exPlanations) to reveal the **true physical factors** influencing energy consumption, providing transparent decision support for operators.

## Core Results and Insights

The model successfully achieved a Mean Absolute Error (MAE) of **11.87%** in predicting energy consumption under the constrained (low-load/compliant) operating conditions.

### 1. Model Performance Summary

| Metric | Value | Comment |
| :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | **32,143.43 kWh** | The average prediction error of the model. If actual operating energy consumption exceeds this margin, it indicates potential operational inefficiency. |
| **MAE as % of Avg. Energy** | $11.87\%$ | The model demonstrates reasonable predictive accuracy. |
| **Average Energy Baseline** | $270,716.17 \text{ kWh}$ | The typical daily energy consumption baseline for the ETP under low-load conditions. |

### 2. Key Energy Drivers Revealed by SHAP

SHAP analysis identified the top three features driving energy consumption under low-load conditions. This informs a layered control strategy:

| Rank | Feature ($\mathbf{X}$) | Physical Mechanism & Operational Advice |
| :--- | :--- | :--- |
| **1** | **Average Temperature** | **Highest influence.** Directly drives microbial activity and reaction rates. **Recommendation:** Implement **seasonal** or **temperature-banded** aeration setpoint adjustments, moving away from fixed year-round setpoints. |
| **2** | **Ammonia** | The primary demand indicator for the nitrification process. **Recommendation:** Shift from traditional DO control to **real-time Ammonia predictive control** to precisely match oxygen supply to nitrification needs, minimizing over-aeration. |
| **3** | **Average Inflow** | Affects hydraulic loading and retention time. **Recommendation:** Utilize storage or equalization basins to **buffer peak flows**. Stabilizing flow minimizes variability in concentration and reduces the energy required for both pumping and aeration control. |

## Project Structure and Technology Stack

### Project Files

* `melbourne_etp_data.csv.csv`: Raw ETP dataset (2014-2019).
* **`melbourne_etp_data_prep.py`**: Data cleaning, handling missing values, and creating the **`Quality_Pass`** constraint label.
* **`train_xgboost_regressor.py`**: Loads optimized data, trains the XGBoost Regressor, calculates MAE/RMSE, and generates SHAP insights and model files (`.joblib`).
* **`etp_dashboard.py`**: Creates the **interactive Web simulator** based on Streamlit (launch with `streamlit run etp_dashboard.py`).

### Technology Stack

* **Python 3.11.7**
* **Machine Learning:** `xgboost`, `scikit-learn`
* **Data Processing:** `pandas`, `numpy`
* **Explainability (XAI):** `shap`
* **Web App/Deployment:** `streamlit`, `joblib` (Model Persistence)

## How to Launch the Interactive Simulator

You can easily run the model interface using Streamlit:

1.  **Environment Setup:**
    ```bash
    pip install pandas numpy scikit-learn xgboost shap joblib streamlit
    ```
2.  **Data & Model Preparation:** Run the prep and training scripts sequentially to generate the optimized training data and the model file.
    ```bash
    python melbourne_etp_data_prep.py
    python train_xgboost_regressor.py
    ```
3.  **Launch Dashboard:**
    ```bash
    streamlit run etp_dashboard.py
    ```

Your browser will open automatically, allowing you to adjust **Average Temperature**, **Ammonia**, and **Average Inflow** sliders to explore their impact on the ETP's **minimum energy consumption**.
