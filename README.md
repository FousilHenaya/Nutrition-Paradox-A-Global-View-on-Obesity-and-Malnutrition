# 🍎 Nutrition Paradox: A Global View on Obesity and Malnutrition

This interactive Streamlit dashboard provides a global overview of obesity and malnutrition trends using WHO datasets. It allows users to explore regional disparities, analyze gender and age group effects, and compare confidence levels across countries. The app bridges the gap between two extremes—over-nutrition and under-nutrition—highlighting the global nutrition paradox.

## 📊 Features

- **Exploratory Data Analysis (EDA)**:
  - Dataset overview with shape, data types, and statistics
  - Distribution plots of `Mean_Estimate` and `CI_Width`
  - Time series trends by region
  - Box plots by region and age group
  - Categorical breakdowns by level
  - Correlation heatmaps and interactive scatter plots

- **SQL-Powered Query Interface**:
  - Select and visualize prebuilt insights like:
    - Top countries/regions by obesity or malnutrition
    - Obesity trends over time
    - Gender and age-based comparisons
    - Confidence interval evaluations
    - Region-level disparities
    - Dual comparison of obesity vs. malnutrition

## 🧪 Tech Stack

- **Frontend**: Streamlit (nutrition.py)
- **Backend**: Python, MySQL (nutritionbackend.py)
- **insights**: word file (nutritioninsights)
- **Visualization**: Matplotlib, Seaborn, Plotly
- **APIs**: WHO Global Health Observatory API
- **Other Libraries**: Pandas, NumPy, PyCountry, Requests

## 🗂️ Data Sources

The data is retrieved from the WHO Global Health Observatory API:

- `https://ghoapi.azureedge.net/api/NCD_BMI_30C` (Adult Obesity)
- `https://ghoapi.azureedge.net/api/NCD_BMI_PLUS2C` (Child Obesity)
- `https://ghoapi.azureedge.net/api/NCD_BMI_18C` (Adult Malnutrition)
- `https://ghoapi.azureedge.net/api/NCD_BMI_MINUS2C` (Child Malnutrition)

Data is filtered to include records from **2012 onwards**.

## 🗂️ MySQL Configuration

- `host = 'localhost'
- `user = 'root'
- `password = 'Shaffie0000'
- `database = 'healthdata'
  
## 🗂️ Run the Streamlit App

- `streamlit run app.py

📸 Preview

💡 Insights

- `Public health research
- `Policy-making support
- `Educational tool for nutrition awareness
- `Identifying underreported countries with high CI_Width



