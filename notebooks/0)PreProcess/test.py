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
# %% DROPPING AFTER THE NULL CHECK
# mech_zipcode shows a lot of missing values so we need to drop it
# Unamed: 0 is just the row index
drop_cols = ['Unnamed: 0', 'merch_zipcode']
df.drop(columns=drop_cols, inplace=True)
print("Dropped columns:", drop_cols)

# %% Data Types
print(set(df.dtypes.to_list()),"\n")
df.info()

# %% Data Description
df.describe()

# %% Duplicate Data
duplicates = df.duplicated().sum()
print(f"Number of duplicates: {duplicates}")

# %% Duplicate Rows
dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}")
if dup_count > 0:
    df.drop_duplicates(inplace=True)
    print("Dropped duplicate rows")


# %% 6) OUTLIER DETECTION (amt example)
#  - Flag txns above 99th percentile
q99 = df['amt'].quantile(0.99)
#creating a new column where encoding -> where 1 indicates high value transaction outlier and 0 indicates normal transaction
df['amt_outlier'] = (df['amt'] > q99).astype(int)
#99% of all transactions are below this value which is $545.99 so only 1 % of transactions are higher than this
# might pass this encoding into a model to predict fraud (behaviour at large spending values)
print(f"99th percentile of amt = {q99:.2f}; flagged amt_outlier")
