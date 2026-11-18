# DeliveryIQ — Delivery Time Prediction System

**Course:** AI Applications  
**Section:** 413 B  
**Team:** Lavish (26BCS11757), Naseeb (26BCS11720), Sangam Saini (26BCS11705), Aryan Kumar (26BCS11744)

A clean Streamlit web application demonstrating dynamic data ingestion, exploratory data analysis (EDA), **Random Forest Regression** training, interactive delivery-time prediction, and holdout model evaluation.

---

## 🌟 Key Features
- **Algorithm — Random Forest Regressor:** Uses an ensemble of 100 decision trees to capture non-linear relationships without manual feature transformation.
- **Dynamic Multi-Column Handling:** Automatically detects the target column, excludes ID tracking columns, and segments features into numeric vs. categorical types.
- **Clean Scikit-Learn Pipeline:** Missing values are imputed gracefully (median for numeric, mode for categorical) inside `ColumnTransformer`. No rows are thrown away!
- **Interactive Prediction Form:** Form sliders and dropdowns adapt automatically to the active dataset columns.
- **Feature Importance Analysis:** Quantifies exactly what percentage each feature contributes toward delivery duration.
- **Holdout Evaluation:** Displays test set MAE, RMSE, R² score, and an Actual vs. Predicted scatter plot.

---

## 📁 Active Dataset Schema (`data/Food_Delivery_Times.csv`)

| Column Name | Type | Description |
|---|---|---|
| `Order_ID` | Identifier | Unique order ID (automatically excluded from features) |
| `Distance_km` | Numeric Feature | Delivery distance in kilometers |
| `Weather` | Categorical Feature | `Clear`, `Rainy`, `Foggy`, `Snowy`, `Windy` |
| `Traffic_Level` | Categorical Feature | `Low`, `Medium`, `High` |
| `Time_of_Day` | Categorical Feature | `Morning`, `Afternoon`, `Evening`, `Night` |
| `Vehicle_Type` | Categorical Feature | `Bike`, `Scooter`, `Car` |
| `Preparation_Time_min` | Numeric Feature | Food preparation time in minutes |
| `Courier_Experience_yrs` | Numeric Feature | Courier experience in years |
| `Delivery_Time_min` | **Target Variable** | Total delivery time in minutes |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
python -m streamlit run app.py
```

---

## 📚 Syllabus Mapping
- **Experiment 1:** Data exploration, Pandas/NumPy preprocessing, and dynamic schema detection.
- **Experiment 2:** EDA, descriptive statistics, distributions, and correlation analysis.
- **Experiment 3:** Supervised ensemble machine learning using Random Forest Regression.
- **Experiment 4:** Model evaluation (MAE, RMSE, R²), feature importance quantification, and interactive inference.
