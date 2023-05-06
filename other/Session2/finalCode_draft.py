###
#
# This code is a draft, based on the notebook for Session 2. It produces only one dataset and skips Tasks 3 and 6.
#
###

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
db = pd.read_csv('../../data/adult.csv')
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

# Apply one-hot-encoding to 6 nominal columns
categorical_cols = df[['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender']].columns
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

# Copy the DataFrame to prepare Outlier Handling
df_old = df.copy()

# Winsorize "hours-per-week"
df['hours-per-week'] = winsorize(df_old['hours-per-week'], limits=(0.09, 0.03))

# Create bins for "age"
minAge = df['age'].min()
maxAge = df['age'].max()
bins = np.linspace(minAge,maxAge,4)

# Define labels for the bins
labels = [0, 1, 2]

# Convert continuous variable "age" into labels 0, 1 and 2 (This is the actual Binning step)
df['age'] = pd.cut(df_old['age'], bins=bins, labels=labels, include_lowest=True)

# Undersample the dataset
X = df.drop('income', axis=1)
y = df['income']
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

# Create new CSV file
df = pd.DataFrame(resampled_data)
df.to_csv('adult_improv.csv', index=False)