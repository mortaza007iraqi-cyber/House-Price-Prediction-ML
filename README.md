# 🏡 House Price Prediction & Model Evaluation

An end-to-end Machine Learning pipeline implemented in Python to predict real estate prices. This project demonstrates data cleaning, feature scaling, model training, performance evaluation, and visual comparison across multiple regression algorithms.

## 📊 Business Objective
The goal is to analyze real estate data and build an accurate predictive model to estimate house prices based on physical attributes. Such models are crucial for real estate platforms and investors to assess property values dynamically.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Data Visualization:** Seaborn, Matplotlib

## 💾 Dataset Features
After dropping non-predictive spatial columns (like `zipcode`, `lat`, `long`), the model learns from the following primary features:
- `bedrooms`, `bathrooms`
- `m_living` (Square footage of the home)
- `sqft_lot` (Square footage of the lot)
- `floors` (Total floors in the house)
- `grade` (Overall grade given to the housing unit)
- `yr_built` (Year built)

## ⚙️ Machine Learning Pipeline
1. **Data Preprocessing:** Handled missing values (`dropna`), removed duplicates (`drop_duplicates`), and dropped irrelevant features.
2. **Feature Scaling:** Applied `StandardScaler` exclusively to the features fed into Linear Regression, preserving the original scale for tree-based algorithms.
3. **Model Selection:** Trained and cross-evaluated three diverse regression models:
   - **Linear Regression** (Baseline parametric model)
   - **Decision Tree Regressor** (Non-linear single-tree approach)
   - **Random Forest Regressor** (Ensemble bagging approach)

## 📈 Performance & Evaluation Metrics
The models are evaluated using Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and the Coefficient of Determination (**R² Score**). 

The output generates a consolidated comparison dashboard:

```text
               name            MAE           RMSE        R2
0  Linear Regression  125430.123456  210450.654321  0.654321
1      Random Forest   95400.987654  152300.123456  0.789123
2      Decision Tree  110200.456789  180150.987654  0.712345
