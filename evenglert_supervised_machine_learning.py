# ==========================================================================================
# Classical Machine Learning - Project: Data Analysis with Supervised Machine Learning
# Author: Evgeniya Englert
# Last update: 2026-01-11
#
# ==========================================================================================
# Objective
# The objective of this project is to develop a machine learning model that can
# perform a data analysis on synthetically generated data.
# You will be required to use one or more common supervised machine learning methods to
# perform the analysis.
#
# Data
# The data that will be used for this project will be synthetically generated.
# The data will consist of various features that may or may not be relevant to the target variable.
# The Python code below is responsible for generating the data you are supposed to work with.
#
# Data Processing
# You are required to perform data cleaning and dimensionality reduction on the data set.
# Data cleaning will involve removing any missing or irrelevant data, and converting the data into a format that can be used by the machine learning algorithm.
# Dimensionality reduction will be performed using PCA (Principal Component Analysis), which is a common technique for reducing the number of features in a data set while retaining the most important information.
#
# Methodology
# You have to apply supervised machine learning methods on the data set at hand.
# This may include regression, decision trees, support vector machines, or other algorithms.
# Evaluate the performance of the model you used by comparing the predictions made by the model to the actual data.
#
# Report of Machine Learning Analysis
# This report summarizes the results of a machine learning analysis to predict a target variable using five regression models:
# Linear Regression, Regression Tree, Random Forest, XGBoost, and Support Vector Machine.
#
# We optimized each model's hyperparameters using RandomizedSearchCV and evaluated their performance through cross-validation.
#
# Model Performance:
# Linear Regression emerged as the best-performing model, demonstrating superior predictive accuracy and a strong fit to the data.
# Model Fit: The model explained 88% of the variance in the target variable (R2=0.88), indicating the strongest relationship with the data compared to the other models.
# Predictive Accuracy: Linear Regression had the lowest Mean Absolute Percentage Error (MAPE) at 73%, signifying an average prediction error of 73% from the actual values. This level of accuracy outperformed all other models in the cross-validation step.
#
# Feature Importance:
# Permutation importance analysis revealed that feature3 was the most influential variable across all models, with feature5 being the second most important.
#
# Additionally, for models evaluation, we used the following metrics: R2, MAPE, MSE, RMSE, MAE.
# ==========================================================================================


# Code
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats # Test for normal distribution
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    make_scorer
)
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import warnings

# Suppress common scikit-learn warnings that can occur with pipelines and search
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Functions ---
# Create a function to generate a dataframe with synthetic data: features and y as target
def generate_synthetic_data(num_samples, num_features):
    """
    Generates a synthetic DataFrame with specified number of samples and features.
    
    Args:
        num_samples (int): The number of samples to generate.
        num_features (int): The number of features to generate.
    
    Returns:
        pd.DataFrame: The generated DataFrame with features and a target 'y'.
    """
    np.random.seed(42)  # for reproducibility
    # Generate random data with normal distribution
    X = np.random.normal(size=(num_samples, num_features))
    # Generate target variable y with a linear combination of a subset of features plus noise
    relevant_features = np.random.choice(num_features, size=int(num_features/2), replace=False)
    w = np.zeros(num_features)
    w[relevant_features] = np.random.normal(size=(len(relevant_features),))
    noise = np.random.normal(scale=0.1, size=(num_samples,))
    y = X[:, relevant_features].dot(w[relevant_features]) + noise
    
    # Introduce missing values in the data
    missing = np.random.choice(num_samples * num_features,
    size=int(num_samples * num_features * 0.1), replace=False)
    X.flat[missing] = np.nan

    # Create a list of feature names
    feature_names = [f'feature{i+1}' for i in range(X.shape[1])]

    # Convert the NumPy array X into a Pandas DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    # Add the 'y' array as a new column to the DataFrame
    df['y'] = y

    return df

# Create a function to check a data frame for missings and print a message
def fct_print_missings_check(df, df_name="DataFrame"):
    """
    Checks a DataFrame for any missing values and prints a status message.
    It prints only the columns that contain missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        df_name (str): The name of the DataFrame for the output message.
    """
    # Sum the missing values for each column
    missing_counts = df.isnull().sum()
    
    # Filter to get only the columns with more than 0 missing values
    columns_with_missings = missing_counts[missing_counts > 0]
    
    # Get the total number of missing values across the entire DataFrame
    total_missing_count = columns_with_missings.sum()
    
    if total_missing_count > 0:
        print(f"❗ Check failed for '{df_name}': The DataFrame has {total_missing_count} missing value(s).")
        print(f" Columns with missings: \n{columns_with_missings}")
    else:
        print(f"✅ Check passed for '{df_name}': The DataFrame has no missing values.")

# --- Data Generation and Exploration ---
print("--- Data Generation and Exploration ---")
num_samples = 1000
num_features = 5
df = generate_synthetic_data(num_samples, num_features)
print("X with feature names and 'y' as a target column (first 5 rows):")
print(df.head())

print(f"\nExplore generated synthetic data, X: {df.shape}")
# Check data for missings
fct_print_missings_check(df, 'df')
print("\nInsight: All features in the original data set have missings and should be modified in a preprocessing step.")

# Data types: all columns are float64
print("\nData types:")
print(df.dtypes)
# Split data into X (features) and y (target)
X = df.drop('y', axis=1)
y = df['y']
# Create a list with feature names
feature_names = pd.DataFrame(df.drop('y', axis=1)).columns
# Plot y with a histogram and with normal distribution:
# y looks normally distributed
sns.histplot(y, bins=10, color='skyblue', stat='density')

# Generate the x-values for the normal distribution curve
x = np.linspace(min(y), max(y), 100)

# Calculate the corresponding y-values using the normal PDF
# Calculate the mean and standard deviation of the data
mu, sigma = np.mean(y), np.std(y)
p = stats.norm.pdf(x, mu, sigma)

# Plot the normal distribution curve
plt.plot(x, p, 'k', linewidth=2)

# Add labels and title
plt.xlabel("Value of the target Y")
plt.ylabel("Density")
plt.title("Histogram with Fitted Normal Distribution")

plt.show()
print("\nInsight: Target variable Y looks normally distributed. Let's proof this hypothesis with the Shapiro-Wilk test for normality.")
# Testing y for normal distribution
# The Shapiro-Wilk test is a formal statistical test for normality.
# The null hypothesis (H0) is that the data is normally distributed.
# A p-value greater than a significance level (e.g., 0.05) suggests we cannot reject H0.
stat, p = stats.shapiro(y)
print(f"Shapiro-Wilk Test Statistic: {stat:.4f}")
print(f"P-value: {p:.4f}")

# You can then interpret the results.
if p > 0.05:
    print("The null hypothesis (H0) that the data is normally distributed can not be rejected. \nThe data likely follows a normal distribution.")
else:
    print("The null hypothesis (H0) that the data is normally distributed can be rejected. \nThe data does not appear to follow a normal distribution.")

# Split data into training and test data sets before handling missings to avoid data leakage.
# 80% for training, 20% for the test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nOriginal data shape: {X.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# --- Methodology and Model Evaluation ---
print("\n" + "=" * 50)
print("--- Machine Learning Analysis ---")

# --- 1. Define the common pipeline steps (pre-processing) ---
preprocessor = Pipeline(steps=[ 
    ('imputer', SimpleImputer(strategy='mean')),  # Step 1: Handle missing values with mean
    ('scaler', StandardScaler())                  # Step 2: Standardize features
])

# --- 2. Define the models to be tested ---
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    'Support Vector Machine': SVR() # Add SVR here
}

# --- 3. Define the hyperparameter search spaces ---
# NOTE: The parameter names must be prefixed with the pipeline step name followed by two underscores.
param_dist = {
    'Linear Regression': {},  # No hyperparameters to tune
    'Decision Tree': {
        'regressor__max_depth': randint(1, 20),
        'regressor__min_samples_split': randint(2, 10),
        'regressor__min_samples_leaf': randint(1, 10),
        'regressor__max_features': ['sqrt', 'log2', None],
    },
    'Random Forest': {
        'regressor__n_estimators': randint(50, 200),
        'regressor__max_depth': randint(3, 20),
        'regressor__min_samples_split': randint(2, 10),
        'regressor__min_samples_leaf': randint(1, 10),
        'regressor__max_features': ['sqrt', 'log2', None],
    },
    'XGBoost': {
        'regressor__n_estimators': randint(50, 500),
        'regressor__max_depth': randint(3, 10),
        'regressor__learning_rate': uniform(0.01, 0.3),
        'regressor__subsample': uniform(0.5, 0.5),
        'regressor__colsample_bytree': uniform(0.5, 0.5)
    },
    'Support Vector Machine': { # Add SVR search space
        'regressor__C': uniform(loc=0.1, scale=100),
        'regressor__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'regressor__gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 10)),
        'regressor__epsilon': uniform(loc=0.01, scale=1)
    }
}

# Dictionary to store the best models for importance calculation later
best_models = {}

# --- 4. Define a custom MAPE function and scorer ---
def mean_absolute_percentage_error_custom(y_true, y_pred):
    """Calculates MAPE, handling zero values in y_true."""
    # Add a small epsilon to the denominator to prevent division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

mape_scorer = make_scorer(mean_absolute_percentage_error_custom, greater_is_better=False)

# --- 5. Iterate through models, perform tuning/CV, and evaluate ---
results = {
    'Model': [],
    'R2 (CV)': [],
    'MAE (CV)': [],
    'RMSE (CV)': [],
    'MAPE (CV)': [],
    'R2 (Test)': [],
    'MAE (Test)': [],
    'RMSE (Test)': [],
    'MAPE (Test)': []
}

for name, regressor in models.items():
    print(f"--- Evaluating {name} ---")

    # Create a full pipeline including pre-processing, dimensionality reduction (for SVR), and the regressor
    if name == 'Support Vector Machine':
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=2)), # Apply PCA for SVR as per your report
            ('regressor', regressor)
        ])
    else:
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

    # Define the scoring metrics for cross-validation
    scoring_metrics = {
        'r2': 'r2',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_mse': 'neg_mean_squared_error',
        'mape': mape_scorer
    }
    
    # Use RandomizedSearchCV for all models, even those with no params to tune,
    # to maintain a consistent workflow and get cross-validated scores.
    model_search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_dist[name],
        n_iter=50 if len(param_dist[name]) > 0 else 1, # Run 50 iterations if params exist, otherwise 1
        cv=5,
        verbose=0,
        random_state=42,
        n_jobs=-1,
        scoring=scoring_metrics,
        refit='r2'  # Refit the best model on the whole training set
    )
    
    # Fit the search object to the data (this performs CV internally)
    model_search.fit(X_train, y_train)
    
    # Get the best estimator from the search
    best_model = model_search.best_estimator_

    # Get the best parameters found by the search.
    print("Best parameters found: ", model_search.best_params_)

    # Extract cross-validation results from the search object
    cv_results = pd.DataFrame(model_search.cv_results_)
    
    # Find the index of the best run
    best_index = model_search.best_index_
    
    # Extract the scores from the best run
    r2_cv = cv_results.loc[best_index, 'mean_test_r2']
    mae_cv = -cv_results.loc[best_index, 'mean_test_neg_mae']
    rmse_cv = np.sqrt(-cv_results.loc[best_index, 'mean_test_neg_mse'])
    mape_cv = -cv_results.loc[best_index, 'mean_test_mape']
    
    # Store the best model for later importance calculation
    best_models[name] = best_model
    
    # --- Evaluate on the test set using the final best model ---
    y_pred = best_model.predict(X_test)

    # Calculate test set metrics
    r2_test = r2_score(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_test = mean_absolute_percentage_error_custom(y_test, y_pred)

    # Store cross-validation and test results
    results['Model'].append(name)
    results['R2 (CV)'].append(r2_cv)
    results['MAE (CV)'].append(mae_cv)
    results['RMSE (CV)'].append(rmse_cv)
    results['MAPE (CV)'].append(mape_cv)
    
    results['R2 (Test)'].append(r2_test)
    results['MAE (Test)'].append(mae_test)
    results['RMSE (Test)'].append(rmse_test)
    results['MAPE (Test)'].append(mape_test)

    # Display results for current model
    print(f"R-squared (CV): {r2_cv:.4f}")
    print(f"MAE (CV): {mae_cv:.4f}")
    print(f"RMSE (CV): {rmse_cv:.4f}")
    print(f"MAPE (CV): {mape_cv:.4f}%")
    print(f"R-squared (Test): {r2_test:.4f}")
    print(f"MAE (Test): {mae_test:.4f}")
    print(f"RMSE (Test): {rmse_test:.4f}")
    print(f"MAPE (Test): {mape_test:.4f}%")
    print("=" * 50 + "\n")


# --- Overall Summary Report: Print and Plot the results ---
print("\n" + "=" * 50)
print("--- Overall Summary Report of Mean Scores ---")
results_df = pd.DataFrame(results)
results_df.set_index('Model', inplace=True)
print(results_df)

# Plotting the results for cross-validation and test performance
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison (Cross-Validation and Test)', fontsize=16)

# Plot R2 scores (CV vs Test)
results_df[['R2 (CV)', 'R2 (Test)']].plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'orange'])
axes[0, 0].set_title('R-squared (R2)')
axes[0, 0].set_ylabel('Score')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot MAE scores (CV vs Test)
results_df[['MAE (CV)', 'MAE (Test)']].plot(kind='bar', ax=axes[0, 1], color=['lightgreen', 'red'])
axes[0, 1].set_title('Mean Absolute Error (MAE)')
axes[0, 1].set_ylabel('Score')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

# Plot RMSE scores (CV vs Test)
results_df[['RMSE (CV)', 'RMSE (Test)']].plot(kind='bar', ax=axes[1, 0], color=['lightcoral', 'dodgerblue'])
axes[1, 0].set_title('Root Mean Squared Error (RMSE)')
axes[1, 0].set_ylabel('Score')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot MAPE scores (CV vs Test)
results_df[['MAPE (CV)', 'MAPE (Test)']].plot(kind='bar', ax=axes[1, 1], color=['lightyellow', 'darkviolet'])
axes[1, 1].set_title('Mean Absolute Percentage Error (MAPE)')
axes[1, 1].set_ylabel('Score')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# --- Find the best model based on MAPE ---
print("\n" + "=" * 50)
print("--- Best Model Selection ---")
# The best model will have the lowest MAPE
best_model_name = results_df['MAPE (CV)'].idxmin()
print(f"The best model based on the lowest Mean Absolute Percentage Error (MAPE) in the Cross-Validation step: {best_model_name}")

# --- Permutation Feature Importance for Tree-Based Models ---
print("\n" + "=" * 50)
print("--- Calculating and Plotting Combined Permutation Importance ---")

# Create an empty DataFrame to store all permutation importances
combined_importance_df = pd.DataFrame(index=X_test.columns)

# Iterate over the best tree-based models and calculate importance
# Permutation importance is not applicable to linear models or SVR
for model_name, best_fitted_model in best_models.items():
    if model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        print(f"\nCalculating Permutation Importance for {model_name}...")
        result = permutation_importance(best_fitted_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        
        # Add the mean importance to the combined DataFrame
        combined_importance_df[model_name] = result.importances_mean

# Sort the combined DataFrame by the importance of one of the models (e.g., Random Forest)
if 'Random Forest' in combined_importance_df.columns:
    combined_importance_df = combined_importance_df.sort_values(by='Random Forest', ascending=False)
    print("Combined Permutation Importances:")
    print(combined_importance_df)

    # Visualize the combined permutation importances
    ax = combined_importance_df.plot(kind='barh', figsize=(12, 8), colormap='viridis')
    ax.set_title("Permutation Feature Importance Comparison Across Models")
    ax.set_xlabel("Importance (Mean Decrease in R-squared)")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# --- Summarize Most Important Features
print("\n" + "=" * 50)
print("--- Most Important Features based on Permutation Importance ---")

for model_name in combined_importance_df.columns:
    most_important_feature = combined_importance_df[model_name].idxmax()
    print(f"The most important feature for the {model_name} model is: {most_important_feature}")

# ==========================================================================================