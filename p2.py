import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

mae = []
rmse = []
r2 = []

def evaluate(name, y_true, y_pred):
    mae.append(mean_absolute_error(y_true, y_pred))
    rmse.append(root_mean_squared_error(y_true, y_pred))
    r2.append(r2_score(y_true, y_pred))

data = pd.read_csv("houses.csv")
dcol = ["zipcode", "yr_renovated", "sqft_basement", "sqft_above", "condition", "view", "waterfront", "sqft_living15", "lat", "long"]
data = data.drop(dcol, axis=1)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

features = data[["bedrooms", "bathrooms", "m_living", "sqft_lot", "floors", "grade", "yr_built"]]
target = data["price"]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

linreg = LinearRegression()
ranforreg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
dectreereg = DecisionTreeRegressor(max_depth=5, random_state=42)

scaler = StandardScaler()

# =======================================linear reg===========================
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)
linreg.fit(x_train_sc, y_train)
pred = linreg.predict(x_test_sc)
evaluate("Linear Regression", y_test, pred)

# =======================================forest reg=============================
ranforreg.fit(x_train, y_train)
pred = ranforreg.predict(x_test)
evaluate("Random Forest", y_test, pred)

# ===============================================tree reg========================
dectreereg.fit(x_train, y_train)
pred = dectreereg.predict(x_test)
evaluate("Decision Tree", y_test, pred)

# بناء الجدول الأصلي الخاص بك
df = pd.DataFrame({
    "name": ["Linear Regression", "Random Forest", "Decision Tree"],
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
})
print(df)

# =============================================== Drawing========================
plt.figure(figsize=(8,5))
sns.barplot(x="name", y="R2", data=df,palette="magma")
plt.title("Model Comparison (R² Score)")
plt.xlabel("Algorithms", fontsize=12)
plt.ylabel("R² Score", fontsize=12)

plt.show()
