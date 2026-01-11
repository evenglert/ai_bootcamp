# Evgeniya Englert 
# Last update: 2025-07-20
# CodeLabs Academy
# Homework: Data Exploration
# The objective of this exercise is to perform data exploration using Pandas and to calculate various statistical measures for a real dataset. 
## By the end of the exercise, you should be able to:
## Load a dataset into a Pandas DataFrame
## Explore the structure and format of the data
## Calculate measures of central tendency, variability, and shape
## Perform basic data cleaning and preprocessing

# Instructions
# Load the "Iris" dataset into a Pandas DataFrame using the code provided above.
# Display the first 5 rows of the DataFrame using the "head()" method.
# Check the shape of the DataFrame using the "shape" attribute.
# Check the data types of each column using the "dtypes" attribute.
# Check for missing values in the DataFrame using the "isnull()" method and the "sum()" method.
# Calculate the mean, median, and mode for the "sepal length (cm)" column.
# Calculate the range, variance, and standard deviation for the "petal width (cm)" column.
# Calculate the skewness and kurtosis for the "sepal width (cm)" column.
# Count the number of unique values in the "target" column using the "nunique()" method.
# Group the data by the "target" column and calculate the mean for each group using the "groupby()" method and the "mean()" method.

# Begin

# Import packages
import pandas as pd
from sklearn.datasets import load_iris

# Load a dataset into a Pandas DataFrame
# Load the iris dataset
data = load_iris() 
# Create a dataframe from the dataset
df = pd.DataFrame(data["data"], columns=data["feature_names"])
df["target"] = data["target"] 
df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"}) 

# Explore the structure and format of the data
# Display the first 5 rows of the DataFrame using the "head()" method.
df.head(5)
# Check the shape of the DataFrame using the "shape" attribute.
df.shape
# Check the data types of each column using the "dtypes" attribute.
df.dtypes
# Check for missing values in the DataFrame using the "isnull()" method and the "sum()" method.
df.isnull().sum()
# Additionally: Check the data structure and format
df.info()

# Calculate measures of central tendency, variability, and shape
# Calculate the mean, median, and mode for the "sepal length (cm)" column.
# Calculate the mean for "sepal length (cm)"
mean_sepal_length = df['sepal length (cm)'].mean()

# Calculate the median for "sepal length (cm)"
median_sepal_length = df['sepal length (cm)'].median()

# Calculate the mode for "sepal length (cm)"
# The .mode() method can return multiple modes if there are ties,
# so it always returns a Series. If you expect a single mode and want
# a scalar, you might take the first element (e.g., .mode()[0]).
mode_sepal_length = df['sepal length (cm)'].mode()

print(f"Mean of 'sepal length (cm)': {mean_sepal_length:.4f}")
print(f"Median of 'sepal length (cm)': {median_sepal_length:.4f}")
print(f"Mode of 'sepal length (cm)': {mode_sepal_length.tolist()}")

# Calculate the range, variance, and standard deviation for the "petal width (cm)" column.
# Calculate the range for "petal width (cm)"
range_petal_width = df['petal width (cm)'].max() - df['petal width (cm)'].min()

# Calculate the variance for "petal width (cm)"
var_petal_width = df['petal width (cm)'].var()

# Calculate the standard deviation for "petal width (cm)"
std_petal_width = df['petal width (cm)'].std()

print(f"Range of 'petal width (cm)': {range_petal_width:.4f}")
print(f"Variance of 'petal width (cm)': {var_petal_width:.4f}")
print(f"Standard deviation of 'petal width (cm)': {std_petal_width:.4f}")

# Calculate the skewness and kurtosis for the "sepal width (cm)" column.
# Calculate the skewness for the "sepal width (cm)" column.
skew_sepal_width = df['sepal width (cm)'].skew()

# Calculate the kurtosis for "sepal width (cm)"
kurt_sepal_width = df['sepal width (cm)'].kurt()

print(f"Skewness of 'sepal width (cm)': {skew_sepal_width:.4f}")
print(f"Kurtosis of 'sepal width (cm)': {kurt_sepal_width:.4f}")

# Additonally:
# Get a statistical summary of the dataframe
df.describe()
# Mean: Get the mean of each numeric column
numeric_columns_mean = df.select_dtypes(include=['number']).mean()
print(numeric_columns_mean)
# Mode: Get the mode of each numeric column
numeric_columns_mode = df.select_dtypes(include=['number']).mode()
print(numeric_columns_mode)
# Median: Get the median of each numeric column
numeric_columns_median = df.select_dtypes(include=['number']).median()
print(numeric_columns_median)
# Standard Deviation: Get the standard deviation of each numeric column
numeric_columns_std = df.select_dtypes(include=['number']).std()
print(numeric_columns_std)
# Variance: Get the variance of each numeric column
numeric_columns_var = df.select_dtypes(include=['number']).std()
print(numeric_columns_var)
# Skewness: Get the skewness of each numeric column
numeric_columns_skew = df.select_dtypes(include=['number']).skew()
print(numeric_columns_skew)
# Kurtosis: Get the kurtosis of each numeric column
numeric_columns_kurt = df.select_dtypes(include=['number']).kurt()
print(numeric_columns_kurt)

# Perform basic data cleaning and preprocessing
# Count the number of unique values in the "target" column using the "nunique()" method.
unique_species_count = df['target'].nunique()
print(f"Number of unique species: {unique_species_count}")

# Group the data by the "target" column and calculate the mean for each group using the "groupby()" method and the "mean()" method.
grouped_means = df.groupby('target').mean()
print("Mean values for each species:")
print(grouped_means)

# Additionally: Check for duplicates in the DataFrame
df.duplicated().sum()
# Insight: One duplicate row found
# Remove duplicate rows
df.drop_duplicates(inplace=True)
# Check the shape of the dataframe after removing duplicates
df.shape # Duplicat was removed

# Additionally:
# % of each species in the target column
species_percentages = df['target'].value_counts(normalize=True) * 100
print("Percentage of each species:")
print(species_percentages)

# End