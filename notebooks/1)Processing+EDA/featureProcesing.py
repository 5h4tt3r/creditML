# %% Feature Processing
'''
In this section we will process the features of the dataset. This includes doing any feature engineering and
transformations being applied to the data. All of this is done after loading in the cleaned version of the data.
'''

# %% Import cleaned csv
import pandas as pd
clean_path = '../../notebooks/1)Processing+EDA/credit_card_transactions_clean.csv'
df = pd.read_csv(clean_path)
# print("Loaded", df.shape, "rows × columns")
df.head(20)

# %% Feature Engineering
# %% Extract Out The Components Of Time
'''
For each transction-date-time we will seperate into hour, weekday, and month
This will help later in anomaly detection and clustering user behaviour based on time of day, day of week, and month.
'''
df['trans_dt'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour']     = df['trans_dt'].dt.hour
df['weekday']  = df['trans_dt'].dt.weekday
df['month']    = df['trans_dt'].dt.month

# (Optional) Drop raw datetime if no longer needed:
print(df.columns.tolist())

# %%dropping the redundant old transaction_date_trans_time column since we have created the new ones above
df.drop(columns=['trans_date_trans_time', 'trans_dt'], inplace=True)
df.head(10)

'''
Now Group By User Using Credit Card Number:
Why do we need this?
A. Time-Series Forecasting
Individual Level

To predict this card’s future spending, you need a history of aggregated values at regular intervals (e.g. daily or weekly sum of amt).
Grouping transforms a long list of transactions into a time series per card:


**Some Feature Engineering Requires them to be grouped by user:
B. Feature Engineering for Classification & Segmentation
Spending Frequency
Count of transactions per card in a window (day/week/month).
Average / Variance
Mean or standard deviation of amt per card.
Category Mix
Fraction of spend in each category per card.
Temporal Patterns (Seasonality)
Distribution of transactions by hour or weekday per card (e.g., entropy or peak times).
All of these must be computed via grouping by cc_num.

Add these features to original dataframe
'''


'''
Some feature engineering will require Non-aggregation
While grouping per user is critical for those features, you still need transaction-level data for:
Merchant-Level Analysis
Fraud rates by merchant or category require looking at each transaction.
Outlier & Anomaly Detection
Rare, extreme transactions (e.g. a single $10,000 purchase) could be diluted if you only look at daily sums.
Sequence Modeling
If you later build models (e.g. RNNs, Transformers) that consume the order of transactions, you must preserve individual records.
'''
