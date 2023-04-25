#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
df = pd.read_csv (r'C:\semster6\DSH\adult.csv')
print (df)


# In[96]:


sp_df = df[:5000]
df = pd.DataFrame(sp_df)
df


# In[97]:


# check for missing values
print(df.isnull().sum())


# In[98]:


# impute missing values with the mean of the column
df.fillna(df.mean(), inplace=True)


# In[99]:


# check again for missing values
print(df.isnull().sum())


# In[100]:



# select only numeric columns
numeric_cols = df.select_dtypes(include='number').columns


# In[101]:


# impute missing values with the mean of numeric columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


# In[102]:



# check again for missing values
print(df.isnull().sum())


# In[103]:


#This output shows that there are no missing values in any of the columns in the 'adult' dataset. 
#Therefore, there is no need to handle missing data in this case.


# In[104]:



# check for noisy data
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    noisy_data = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
    if not noisy_data.empty:
        print(f"Noisy data found in column {col}:")
        print(noisy_data)


# In[105]:


# check for inconsistencies
inconsistent_data = df[(df['capital-gain'] > 0) & (df['capital-loss'] > 0)]
if not inconsistent_data.empty:
    print("Inconsistent data found:")
    print(inconsistent_data)


# In[106]:


# check for duplicate entries
duplicates = df[df.duplicated()]
if not duplicates.empty:
    print("Duplicate entries found:")
    print(duplicates)


# In[107]:


# drop duplicate entries
df.drop_duplicates(inplace=True)


# In[108]:


# check again for missing values
print(df.isnull().sum())


# In[109]:


# one-hot encoding
from sklearn.preprocessing import OneHotEncoder
# Select numeric columns
numeric_cols = df.select_dtypes(include='number').columns

# Select categorical columns
categorical_cols = df.columns.difference(numeric_cols)

# Perform one-hot encoding on categorical columns
onehot_encoder = OneHotEncoder(sparse=False)
onehot_array = onehot_encoder.fit_transform(df[categorical_cols])
column_names = onehot_encoder.get_feature_names_out(categorical_cols)
onehot_df = pd.DataFrame(onehot_array, columns=column_names)

# Concatenate one-hot encoded DataFrame with original numeric columns
numeric_df = df[numeric_cols].reset_index(drop=True)
processed_df = pd.concat([numeric_df, onehot_df], axis=1)

# Verify that all columns are present in the processed DataFrame
print(processed_df.columns)


# In[86]:


#The increase in the number of columns from 15 to 102 after one-hot encoding indicates 
#that there were 87 unique categorical values across the 15 original categorical columns. 
#One-hot encoding created a new binary column for each unique value, resulting in the increase in the number of columns. 
#This is a common outcome of one-hot encoding, and it enables us to represent categorical variables as numerical variables
#in a way that can be easily processed by machine learning algorithms.


# In[111]:


processed_df['age']


# In[112]:


#processed_df['marital-status']
print(processed_df.columns.tolist())


# In[113]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
#To scale the continuous features in the processed_df dataset, 
#we can use either min-max scaling or normalization. 
#Here's an example of how to apply these techniques using scikit-learn:
# Select continuous columns
continuous_cols = processed_df.select_dtypes(include='number').columns


# In[114]:


# Perform min-max scaling
min_max_scaler = MinMaxScaler()
processed_df[continuous_cols] = min_max_scaler.fit_transform(processed_df[continuous_cols])


# In[115]:


# Perform normalization
standard_scaler = StandardScaler()
processed_df[continuous_cols] = standard_scaler.fit_transform(processed_df[continuous_cols])


# In[117]:


print(processed_df[continuous_cols])


# In[119]:


import numpy as np
import pandas as pd
import seaborn as sns

# Compute summary statistics for each feature
summary_stats = processed_df.describe()

# Compute interquartile range (IQR) for each feature
q1 = processed_df.quantile(0.25)
q3 = processed_df.quantile(0.75)
iqr = q3 - q1

# Compute lower and upper bounds for identifying outliers
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Identify potential outliers
outliers = {}
for col in processed_df.columns:
    outliers[col] = processed_df[(processed_df[col] < lower_bound[col]) | (processed_df[col] > upper_bound[col])]


# In[120]:



# Visualize the distribution of each continuous feature
sns.boxplot(data=processed_df[numeric_cols])
sns.histplot(data=processed_df[numeric_cols])


# In[ ]:




