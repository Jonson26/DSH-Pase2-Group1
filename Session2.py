'''This python file produces 6 CSV files by performing the following steps:

 Task 1 - Cleaning :
 (1) Impute missing values (2) Remove duplicates

 Task 2 - Encoding :
 (1) Drop 1 feature (2) Label-encode class (3) One-hot-encode some features* (4) Target-encode 1 feature

 Task 4 - Outlier handling :
 (1) Winsorize 1 feature (2) Bin 1 other feature

 Task 5 - Balancing :
 (1) Undersample

 Task 6 - Feature Engineering :
 (1) *(before:) Create feature "occupation-type" (2) Make a copy with a simplified "capital" feature and *WITHOUT THE ORIGINAL "CAPITAL" FEATURES*

 Task 3 - Scaling :
 For the two existing copies:
 (1) Make a copy with Normalized values (2) Make a copy with Standardized values
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats.mstats import winsorize
import seaborn as sns
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler

# Read the dataset and create a dataframe
db = pd.read_csv('data/adult.csv')
df = pd.DataFrame(db)


# Impute missing values in variable "workclass" with most frequent category
df.workclass.replace('?','Private', inplace=True)

# Do the same for "occupation"
df.occupation.replace('?','Prof-specialty', inplace=True)

# Do the same for "native-country"
df['native-country'].replace('?','United-States', inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)


# Remove variable "education"
df.drop(columns='education',inplace=True)

# Replace "income" value "<=50K" with 0, and "income" value ">50K" with 1
df['income'].replace('<=50K', 0, inplace=True)
df['income'].replace('>50K', 1, inplace=True)

# Select rows of different "occupation" types and set according value in new feature "occupation-type"
df.loc[df['occupation'].isin(['Prof-specialty', 'Exec-managerial', 'Adm-clerical', 'Farming-fishing', 'Armed-Forces']), 'occupation-type'] = 'Professional'
df.loc[df['occupation'].isin(['Craft-repair', 'Machine-op-inspct', 'Tech-support']), 'occupation-type'] = 'Technical'
df.loc[df['occupation'].isin(['Sales', 'Other-service', 'Transport-moving', 'Handlers-cleaners', 'Protective-serv', 'Priv-house-serv']), 'occupation-type'] = 'Service'

# Apply one-hot-encoding to categorical columns
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'occupation-type']
numeric_cols = df.columns.difference(categorical_cols)
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_array = onehot_encoder.fit_transform(df[categorical_cols])
column_names = onehot_encoder.get_feature_names_out(categorical_cols)
onehot_df = pd.DataFrame(onehot_array, columns=column_names)
numeric_df = df[numeric_cols].reset_index(drop=True)
encoded_df = pd.concat([numeric_df, onehot_df], axis=1)
df = encoded_df.copy()

# Target-Encode "native-country"
target_encoder = TargetEncoder()
df['native-country'] = target_encoder.fit_transform(encoded_df['native-country'], encoded_df['income'])


# Backup the DataFrame to prepare Outlier Handling
df_old = df.copy()

# Winsorize "hours-per-week"
df['hours-per-week'] = winsorize(df_old['hours-per-week'], limits=(0.09, 0.03))

# Create bins for "age"
minAge = df['age'].min()
maxAge = df['age'].max()
bins = [17, 25, 50, 90]

# Define labels for the bins
labels = [0, 1, 2]

# Use the bins to create age groups (This is the actual Binning step)
df['age-group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)


# Undersample the dataset
X = df.drop('income', axis=1)
y = df['income']
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
df = pd.DataFrame(resampled_data)


# Make a copy from which the original capital features will be removed
df_nocap = df.copy()

# Create "capital" as difference between gain and loss
df_nocap['capital'] = df['capital-gain'] - df['capital-loss']

# Create bins for "capital"
minCap = df_nocap['capital'].min()
maxCap = df_nocap['capital'].max()
binsCap = [minCap, -1, 1, maxCap]

# Use the bins to create capital groups
df_nocap['capital'] = pd.cut(df_nocap['capital'], bins=binsCap, labels=labels, include_lowest=True)

# Remove variables "capital-gain" and "capital-loss"
df_nocap.drop(columns=['capital-gain', 'capital-loss'],inplace=True)


# Select columns for scaling
continuous_cols = ['age', 'capital-gain', 'capital-loss', 'educational-num', 'fnlwgt', 'hours-per-week']
continuous_cols_nocap = ['age', 'educational-num', 'fnlwgt', 'hours-per-week']

# Create normalized versions of the two copies
min_max_scaler = MinMaxScaler()
minmax_df = df.copy()
minmax_df_nocap = df_nocap.copy()
minmax_df[continuous_cols] = min_max_scaler.fit_transform(minmax_df[continuous_cols])
minmax_df_nocap[continuous_cols_nocap] = min_max_scaler.fit_transform(minmax_df_nocap[continuous_cols_nocap])

# Create standardized versions of the two copies
standard_scaler = StandardScaler()
standard_df = df.copy()
standard_df_nocap = df_nocap.copy()
standard_df[continuous_cols] = standard_scaler.fit_transform(standard_df[continuous_cols])
standard_df_nocap[continuous_cols_nocap] = standard_scaler.fit_transform(standard_df_nocap[continuous_cols_nocap])


# Create CSV for each copy of the dataframe
df.to_csv('preprocessed.csv', index=False)
minmax_df.to_csv('preprocessed_normalized.csv', index=False)
standard_df.to_csv('preprocessed_standardized.csv', index=False)
df_nocap.to_csv('preprocessed_noCapital.csv')
minmax_df_nocap.to_csv('preprocessed_noCapital_normalized.csv', index=False)
standard_df_nocap.to_csv('preprocessed_noCapital_standardized.csv', index=False)