# %% IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import os
style.use('ggplot')

# %%check current working directory
print("current working directory:", os.getcwd())

# %% Load Train.csv into DataFrame
# Load Train.csv into DataFrame
dataPath = '../../data/credit_card_transactions.csv'
df = pd.read_csv(dataPath)
df.head(20)

# %% DATA CLEANING CHECKLIST
# Check for missing values
# Check data types
# Check for duplicates
# Check for outliers
# Check for data type consistency
# Check for data imbalances

# %% Data For Missing Values
df.isnull().sum().sort_values(ascending=False)

# %% Data Types
print(set(df.dtypes.to_list()),"\n")
df.info()

# %% Data Description
df.describe()

# %% Duplicate Data
duplicates = df.duplicated().sum()
print(f"Number of duplicates: {duplicates}")
df['Time'].describe()
df['Time'].value_counts()
df['Time'].value_counts().plot(kind='bar')
df['Time'].value_counts().plot(kind='bar').set_title('Time Distribution')
df['Time'].value_counts().plot(kind='bar').set_title('Time Distribution').set_xlabel('Time').set_ylabel('Count')
