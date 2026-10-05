# House Price Prediction Using Linear Regression

## Project Overview

The objective of this project is to understand and implement Linear Regression for a real-world prediction problem.
The project uses a housing dataset containing information such as:
- House area
- Number of bedrooms
- Number of bathrooms
- Number of stories
- Parking spaces
- Main road access
- Guest room availability
- Basement availability
- Air conditioning
- Hot water heating
- Preferred area
- Furnishing status

The target variable is **house price**.
Two regression approaches were implemented:
1. **Simple Linear Regression** – uses house area to predict price.
2. **Multiple Linear Regression** – uses multiple property features to predict price.

## Objectives

- Import and preprocess the housing dataset.
- Explore relationships between features and house prices.
- Split the dataset into training and testing sets.
- Implement Simple Linear Regression.
- Implement Multiple Linear Regression.
- Evaluate models using MAE, MSE, RMSE and R².
- Visualize the regression line and predictions.
- Interpret regression coefficients.
- Compare the performance of simple and multiple regression.

## Technologies Used

- **Python**
- **Pandas** – data loading and preprocessing
- **NumPy** – numerical operations
- **Matplotlib** – visualization
- **Seaborn** – statistical visualization
- **Scikit-learn** – machine learning and evaluation

## Dataset

The dataset contains **545 house records and 13 columns**.

### Target Variable

`price`
The variable represents the price of the house.

### Features

| Feature | Description |
|---|---|
| `area` | Area of the house |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether the house is connected to the main road |
| `guestroom` | Availability of a guest room |
| `basement` | Availability of a basement |
| `hotwaterheating` | Availability of hot water heating |
| `airconditioning` | Availability of air conditioning |
| `parking` | Number of parking spaces |
| `prefarea` | Whether the house is in a preferred area |
| `furnishingstatus` | Furnishing condition of the house |

# Project Workflow

Dataset
  ↓
Data Inspection
  ↓
Data Cleaning & Preprocessing
  ↓
Exploratory Data Analysis
  ↓
Train-Test Split
  ↓
Linear Regression Models
  ↓
Predictions
  ↓
Model Evaluation
  ↓
Visualization & Analysis
  ↓
Model Comparison
  ↓
Final Results

## Working

- Loaded and explored a housing dataset containing **545 records and 13 columns**.
- Checked the dataset for missing values and duplicates.
- Performed data preprocessing, including handling numerical and categorical features.
- Converted categorical variables into numerical features using **one-hot encoding**.
- Split the dataset into **80% training and 20% testing data**.
- Built a **Simple Linear Regression** model using `area` as the predictor.
- Built a **Multiple Linear Regression** model using multiple property-related features.
- Evaluated both models using **MAE, MSE, RMSE and R²**.
- Created visualizations including a price distribution, correlation heatmap, regression line, actual vs predicted prices, residual plot and coefficient analysis.
- Interpreted the regression coefficients to understand how different features influence predicted house prices.
- Generated a sample prediction using the trained Multiple Linear Regression model.

## Simple Regression Equation

The Simple Linear Regression model produced the following equation:
Price = 2,512,254.26 + (425.73 × area)

The `area` coefficient indicates that a one-unit increase in area is associated with an estimated **425.73-unit increase in predicted house price** in the simple regression model.

## Model Results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Simple Regression | 1,474,748.13 | 3.68 × 10¹² | 1,917,103.70 | 0.2729 |
| **Multiple Regression** | **970,043.40** | **1.75 × 10¹²** | **1,324,506.96** | **0.6529** |

The **Multiple Linear Regression model performed better**, achieving an R² score of **0.6529** and lower prediction errors across all three error metrics.

## Outputs

The project generates:
- Model comparison report
- Feature coefficient report
- Prediction results
- Price distribution plot
- Correlation heatmap
- Regression line
- Actual vs predicted plot
- Residual plot
- Feature coefficient visualization