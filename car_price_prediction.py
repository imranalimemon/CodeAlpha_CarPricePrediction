"""
Car Price Prediction — CodeAlpha Data Science Internship (Task 3)
Author: Imran Ali Memon
Description: Predict car prices using regression models based on 
             features like brand, horsepower, mileage, engine size, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
RANDOM_STATE = 42
FIGURE_DIR = "figures"

import os
os.makedirs(FIGURE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. CREATE / LOAD DATASET
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("CAR PRICE PREDICTION")
print("=" * 60)

# Generate a realistic car dataset
np.random.seed(RANDOM_STATE)
n = 500

brands = np.random.choice(["Toyota", "Honda", "BMW", "Mercedes", "Ford", "Hyundai", "Audi", "Kia"], n)
fuel_types = np.random.choice(["Petrol", "Diesel", "CNG", "Electric"], n, p=[0.45, 0.30, 0.15, 0.10])
transmissions = np.random.choice(["Manual", "Automatic"], n, p=[0.55, 0.45])

year = np.random.randint(2005, 2025, n)
age = 2026 - year
km_driven = np.random.randint(5000, 200000, n)
engine_cc = np.random.choice([800, 1000, 1200, 1500, 1800, 2000, 2500, 3000], n)
max_power_bhp = engine_cc * 0.06 + np.random.normal(0, 8, n)
mileage_kmpl = 25 - engine_cc * 0.005 + np.random.normal(0, 2, n)
seats = np.random.choice([4, 5, 6, 7], n, p=[0.1, 0.6, 0.15, 0.15])

# Price formula (realistic-ish)
brand_premium = {"Toyota": 1.1, "Honda": 1.0, "BMW": 2.5, "Mercedes": 2.8,
                 "Ford": 0.9, "Hyundai": 0.85, "Audi": 2.3, "Kia": 0.8}
brand_mult = np.array([brand_premium[b] for b in brands])
fuel_mult = np.where(fuel_types == "Electric", 1.3, np.where(fuel_types == "Diesel", 1.1, 1.0))
trans_mult = np.where(transmissions == "Automatic", 1.15, 1.0)

price = (
    engine_cc * 3.5
    + max_power_bhp * 20
    - age * 400
    - km_driven * 0.02
    + mileage_kmpl * 50
) * brand_mult * fuel_mult * trans_mult + np.random.normal(0, 50000, n)
price = np.clip(price, 80000, 5000000).astype(int)

df = pd.DataFrame({
    "Brand": brands, "Year": year, "Fuel_Type": fuel_types,
    "Transmission": transmissions, "Engine_CC": engine_cc,
    "Max_Power_BHP": np.round(max_power_bhp, 1),
    "Mileage_KMPL": np.round(mileage_kmpl, 1),
    "KM_Driven": km_driven, "Seats": seats, "Price": price,
})

# Save dataset
df.to_csv("car_dataset.csv", index=False)
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n🔍 First 5 Rows:")
print(df.head())
print(f"\n📈 Statistical Summary:")
print(df.describe())
print(f"\n❌ Missing Values: {df.isnull().sum().sum()}")

# ═══════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 2a. Price Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["Price"], bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
axes[0].set_title("Price Distribution", fontweight="bold")
axes[0].set_xlabel("Price (₹)")
axes[1].hist(np.log1p(df["Price"]), bins=30, color="#FF9800", edgecolor="white", alpha=0.8)
axes[1].set_title("Log-Transformed Price", fontweight="bold")
axes[1].set_xlabel("Log(Price)")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/01_price_distribution.png", dpi=150)
plt.close()
print("✅ Saved: 01_price_distribution.png")

# 2b. Price by Brand
plt.figure(figsize=(10, 6))
brand_order = df.groupby("Brand")["Price"].median().sort_values(ascending=False).index
sns.boxplot(x="Brand", y="Price", data=df, order=brand_order, palette="Set2")
plt.title("Price Distribution by Brand", fontsize=14, fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/02_price_by_brand.png", dpi=150)
plt.close()
print("✅ Saved: 02_price_by_brand.png")

# 2c. Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, cmap="RdYlBu_r", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()
print("✅ Saved: 03_correlation_heatmap.png")

# 2d. Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = ["Engine_CC", "Max_Power_BHP", "KM_Driven", "Year"]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
for i, (feat, c) in enumerate(zip(features, colors)):
    ax = axes[i // 2, i % 2]
    ax.scatter(df[feat], df["Price"], alpha=0.4, s=20, color=c)
    ax.set_xlabel(feat)
    ax.set_ylabel("Price")
    ax.set_title(f"Price vs {feat}", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/04_scatter_plots.png", dpi=150)
plt.close()
print("✅ Saved: 04_scatter_plots.png")

# ═══════════════════════════════════════════════════════════
# 3. DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

df_model = df.copy()

# Encode categorical features
le_brand = LabelEncoder()
le_fuel = LabelEncoder()
le_trans = LabelEncoder()

df_model["Brand"] = le_brand.fit_transform(df_model["Brand"])
df_model["Fuel_Type"] = le_fuel.fit_transform(df_model["Fuel_Type"])
df_model["Transmission"] = le_trans.fit_transform(df_model["Transmission"])

# Feature/Target split
X = df_model.drop("Price", axis=1)
y = df_model["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════
# 4. MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL TRAINING & COMPARISON")
print("=" * 60)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=100),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
    
    results[name] = {"mae": mae, "rmse": rmse, "r2": r2,
                     "cv_mean": cv.mean(), "y_pred": y_pred}
    
    print(f"\n{'─' * 50}")
    print(f"📌 {name}")
    print(f"   MAE:  ₹{mae:,.0f}")
    print(f"   RMSE: ₹{rmse:,.0f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   CV R²: {cv.mean():.4f} ± {cv.std():.4f}")

# ═══════════════════════════════════════════════════════════
# 5. RESULTS VISUALIZATION
# ═══════════════════════════════════════════════════════════

# 5a. R² Comparison
model_names = list(results.keys())
r2_scores = [results[m]["r2"] for m in model_names]

plt.figure(figsize=(10, 6))
bars = plt.barh(model_names, r2_scores, color=["#2196F3", "#4CAF50", "#FF9800", "#e74c3c", "#9b59b6"])
plt.xlabel("R² Score", fontsize=12)
plt.title("Model Comparison — R² Score", fontsize=14, fontweight="bold")
for bar, val in zip(bars, r2_scores):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=10)
plt.xlim(0, 1.1)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/05_model_r2_comparison.png", dpi=150)
plt.close()
print("\n✅ Saved: 05_model_r2_comparison.png")

# 5b. Actual vs Predicted (Best Model)
best_name = max(results, key=lambda k: results[k]["r2"])
best_pred = results[best_name]["y_pred"]

plt.figure(figsize=(8, 8))
plt.scatter(y_test, best_pred, alpha=0.5, s=30, color="#e74c3c")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         "k--", linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual Price (₹)", fontsize=12)
plt.ylabel("Predicted Price (₹)", fontsize=12)
plt.title(f"Actual vs Predicted — {best_name}", fontsize=14, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/06_actual_vs_predicted.png", dpi=150)
plt.close()
print("✅ Saved: 06_actual_vs_predicted.png")

# 5c. Feature Importance (if tree-based model is best)
if hasattr(models[best_name], "feature_importances_"):
    importances = models[best_name].feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    feat_imp.plot(kind="barh", color="#3498db")
    plt.title(f"Feature Importance — {best_name}", fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/07_feature_importance.png", dpi=150)
    plt.close()
    print("✅ Saved: 07_feature_importance.png")

# Summary
print(f"\n🏆 Best Model: {best_name}")
print(f"   R²: {results[best_name]['r2']:.4f}")
print(f"   MAE: ₹{results[best_name]['mae']:,.0f}")

summary_df = pd.DataFrame({
    "Model": model_names,
    "MAE": [f"₹{results[m]['mae']:,.0f}" for m in model_names],
    "RMSE": [f"₹{results[m]['rmse']:,.0f}" for m in model_names],
    "R²": [f"{results[m]['r2']:.4f}" for m in model_names],
})
print("\n📊 Final Comparison:")
print(summary_df.to_string(index=False))
print("\n✅ Car Price Prediction Complete!")
