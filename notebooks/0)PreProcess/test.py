# %% IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import os
from turtledemo.penrose import f
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

# %% 7) CLASS BALANCE
fraud_rate = df['is_fraud'].mean()
print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
# If extremely imbalanced, plan SMOTE / downsampling etc.
counts = df['is_fraud'].value_counts()
rates  = df['is_fraud'].value_counts(normalize=True)
print(pd.DataFrame({'count': counts, 'rate': rates}))

'''
1. Why this matters
Baseline “dumb” classifier

A model that always predicts “not fraud” would be 99.42% accurate.
That sounds great… but it never catches a single fraud.

Misleading accuracy
In imbalanced settings, accuracy is almost worthless.
You need metrics that emphasize the minority (fraud) class.

Choosing the right evaluation metrics
Metric	What it tells you
Precision	Of all transactions flagged as fraud, what fraction truly were fraud?
Recall	Of all actual frauds, what fraction did you catch?
F1-Score	Harmonic mean of precision & recall; balances false positives & negatives.
ROC-AUC	Model’s ability to rank fraud vs non-fraud across all thresholds.
PR-AUC	Area under the Precision-Recall curve; more informative on imbalanced data.

When training will have to split on the fraud column so that train/val/test all have the same fraud rate ~0.58%
This prevents model from seeing different fraud rates in each split.

Algorithmic Approaches
1) Class weights / Cost Sensitive Learning
-Tell the model that mistakes on fraud cost more than on normal transactions
-Anomaly detection which will test fraud as anomalies since train model will learn that normal transactions are not frauds (autoencoders)

Note:
When fraud is rare, PR-AUC is often more meaningful than ROC-AUC, because PR-AUC focuses solely on performance for the positive (fraud) class.
'''



# %% UPLOADING THE CLEANED CSV
clean_path = '../../notebooks/1)Processing+EDA/credit_card_transactions_clean.csv'
df.to_csv(clean_path, index=False)
print("Cleaned data saved to", clean_path)
