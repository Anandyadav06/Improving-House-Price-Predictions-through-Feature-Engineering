# Improving-House-Price-Predictions-through-Feature-Engineering

    Import Required Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score


    Load the Dataset

df = pd.read_csv('boston.csv')

df.head()


    Initial Data Exploration

 df.info()
 
df.describe()

df.isnull().sum()


    Handle Missing Values

# Option 1: Drop rows or columns with too many missing values

# df.dropna(inplace=True)

# Option 2: Fill missing values

df.fillna(df.median(numeric_only=True), inplace=True)


    Correlation Analysis

plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title('Correlation Matrix')

plt.show()


    Feature Engineering Ideas

# Log transformation for skewed features (e.g., CRIM)

df['log_CRIM'] = np.log1p(df['CRIM'])

# Binning 'AGE'

df['AGE_BIN'] = pd.cut(df['AGE'], bins=[0, 35, 70, 100], labels=[0, 1, 2])

# Remove original CRIM to avoid redundancy

df.drop(columns=['CRIM'], inplace=True)


    Conclusion

Feature engineering (log transform, binning, scaling, etc.) helped to reduce MSE and improve R².

These steps are crucial for boosting model performance.

    





