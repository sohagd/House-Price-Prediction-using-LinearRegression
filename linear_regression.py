import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Housing.csv")
print("Dataset Preview:")
print(df.head())
print("\nDataset Information:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()
target = "price"
df[target] = df[target].astype(str).str.replace(
    r"[₹$,]", "", regex=True
)
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])

# Separate features and target
X = df.drop(columns=[target]).dropna(axis=1, how="all")
y = df[target]
if "area" in X.columns:
    feature = "area"
else:
    numeric_features = X.select_dtypes(include=np.number).columns
    if len(numeric_features) == 0:
        raise ValueError("No numerical feature found.")
    feature = numeric_features[0]
X[feature] = pd.to_numeric(X[feature], errors="coerce")
os.makedirs("outputs", exist_ok=True)

# Exploratory Data Analysis (EDA)
print("\nStatistical Summary:")
print(df.describe())

# House Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df[target], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/price_distribution.png")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)
if numeric_df.shape[1] > 1:
    plt.figure(figsize=(9, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.show()

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simple Linear Regression
X_train_simple = X_train[[feature]].copy()
X_test_simple = X_test[[feature]].copy()
area_median = X_train_simple[feature].median()

X_train_simple[feature] = X_train_simple[feature].fillna(area_median)
X_test_simple[feature] = X_test_simple[feature].fillna(area_median)

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)

simple_predictions = simple_model.predict(X_test_simple)

# Multiple Linear Regression
X_train_multi = X_train.copy()
X_test_multi = X_test.copy()

numeric_cols = X_train_multi.select_dtypes(
    include=np.number
).columns.tolist()
categorical_cols = [
    col for col in X_train_multi.columns
    if col not in numeric_cols
]

for col in numeric_cols:
    median_value = X_train_multi[col].median()
    X_train_multi[col] = X_train_multi[col].fillna(median_value)
    X_test_multi[col] = X_test_multi[col].fillna(median_value)

for col in categorical_cols:
    mode_value = X_train_multi[col].mode().iloc[0]
    X_train_multi[col] = X_train_multi[col].fillna(mode_value)
    X_test_multi[col] = X_test_multi[col].fillna(mode_value)
    X_train_multi[col] = X_train_multi[col].astype(str)
    X_test_multi[col] = X_test_multi[col].astype(str)

# Convert categorical features into numerical values
X_train_multi = pd.get_dummies(
    X_train_multi,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)
X_test_multi = pd.get_dummies(
    X_test_multi,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

# Align test features with training features
X_test_multi = X_test_multi.reindex(
    columns=X_train_multi.columns,
    fill_value=0
)

# Train the model
multiple_model = LinearRegression()
multiple_model.fit(X_train_multi, y_train)
multiple_predictions = multiple_model.predict(X_test_multi)

# Model Evaluation
def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    print(f"\n{name}")
    print("-" * 35)
    print(f"MAE  : {mae:,.2f}")
    print(f"MSE  : {mse:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")
    return [mae, mse, rmse, r2]

# Evaluate Simple Regression
simple_results = evaluate_model(
    "Simple Linear Regression",
    y_test,
    simple_predictions
)

# Evaluate Multiple Regression
multiple_results = evaluate_model(
    "Multiple Linear Regression",
    y_test,
    multiple_predictions
)

# 9. Model Comparison
comparison = pd.DataFrame(
    [simple_results, multiple_results],
    columns=["MAE", "MSE", "RMSE", "R2"],
    index=["Simple Regression", "Multiple Regression"]
)
print("\nModel Comparison:")
print(comparison.round(4))
comparison.to_csv("outputs/model_comparison.csv")

# Plotting the Simple Regression Line
plt.figure(figsize=(8, 5))
plt.scatter(
    X_test_simple[feature],
    y_test,
    alpha=0.6,
    label="Actual Prices"
)

# Sort values for a smooth regression line
sorted_indices = np.argsort(X_test_simple[feature].values)
plt.plot(
    X_test_simple[feature].values[sorted_indices],
    simple_predictions[sorted_indices],
    color="red",
    linewidth=2,
    label="Regression Line"
)
plt.xlabel(feature.capitalize())
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/simple_regression.png")
plt.show()

# Actual vs Predicted Prices
plt.figure(figsize=(8, 5))
plt.scatter(
    y_test,
    multiple_predictions,
    alpha=0.6
)
min_value = min(y_test.min(), multiple_predictions.min())
max_value = max(y_test.max(), multiple_predictions.max())
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    label="Perfect Prediction"
)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

# Residual Analysis
residuals = y_test - multiple_predictions
plt.figure(figsize=(8, 5))
plt.scatter(
    multiple_predictions,
    residuals,
    alpha=0.6
)
plt.axhline(y=0, linestyle="--", color="red")
plt.xlabel("Predicted House Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("outputs/residual_plot.png")
plt.show()

# Coefficient Interpretation
# Simple Regression Coefficient
simple_coefficient = simple_model.coef_[0]
simple_intercept = simple_model.intercept_
print("\nSimple Linear Regression Coefficients:")
print(f"Intercept: {simple_intercept:,.2f}")
print(f"{feature} Coefficient: {simple_coefficient:,.2f}")
print(
    f"\nFor every one-unit increase in {feature}, "
    f"the predicted house price changes by "
    f"{simple_coefficient:,.2f} units."
)
print(
    f"\nRegression Equation:\n"
    f"Price = {simple_intercept:,.2f} "
    f"+ ({simple_coefficient:,.2f} * {feature})"
)

# Multiple Regression Coefficients
coefficients = pd.DataFrame({
    "Feature": X_train_multi.columns,
    "Coefficient": multiple_model.coef_
})
coefficients["Absolute Coefficient"] = (
    coefficients["Coefficient"].abs()
)
coefficients = coefficients.sort_values(
    by="Absolute Coefficient",
    ascending=False
)
print("\nMultiple Regression Coefficients:")
print(
    coefficients[["Feature", "Coefficient"]]
    .to_string(index=False)
)
print(f"\nIntercept: {multiple_model.intercept_:,.2f}")

coefficients.to_csv(
    "outputs/feature_coefficients.csv",
    index=False
)

# Plot top 10 coefficients by absolute magnitude
top_features = coefficients.head(10).sort_values(
    by="Coefficient"
)
plt.figure(figsize=(9, 6))
plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)
plt.axvline(x=0, color="black", linewidth=1)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Top 10 Feature Coefficients")
plt.tight_layout()
plt.savefig("outputs/feature_coefficients.png")
plt.show()

# Sample House Price Prediction
sample_house = X_test_multi.iloc[[0]]
predicted_price = multiple_model.predict(sample_house)[0]
actual_price = y_test.iloc[0]
print("\nSample House Price Prediction:")
print(f"Actual Price: {actual_price:,.2f}")
print(f"Predicted Price: {predicted_price:,.2f}")

prediction_results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": multiple_predictions,
    "Residual": residuals.values
})
prediction_results.to_csv(
    "outputs/prediction_results.csv",
    index=False
)

print("\nFinal Model Summary")
print("=" * 40)
print(f"Simple Regression R²: {simple_results[3]:.4f}")
print(f"Multiple Regression R²: {multiple_results[3]:.4f}")
if multiple_results[3] > simple_results[3]:
    print("\nMultiple Linear Regression performed better.")
else:
    print("\nSimple Linear Regression performed better.")