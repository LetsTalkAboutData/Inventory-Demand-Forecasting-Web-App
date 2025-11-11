import pandas as pd

# Load the data
df_train = pd.read_csv('train.csv', parse_dates=['date'])
df_test = pd.read_csv('test.csv', parse_dates=['date'])

# Rename columns for consistency
df_train = df_train.rename(columns={'sales': 'demand'})

# Data Cleaning And Feature Engineering
def create_time_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    return df

df_train = create_time_features(df_train)

# Lag Feature Engineering
def create_lag_features(df, lags=[7, 28]):
    ##Create specified lag features for the demand column.
    df_temp = df.copy()

    ##Sort data for correct lag calculation
    df_temp = df_temp.sort_values(by=['store', 'item', 'date'])

    for lag in lags:
        df_temp[f'lag_{lag}'] = df_temp.groupby(['store', 'item'])['demand'].shift(lag)

    ## Fill NA values with 0
    df_temp = df_temp.fillna(0)

    return df_temp

df_train = create_lag_features(df_train)

# Define the date to split training and validation sets
split_date = '2017-06-01'

# Features to be used for modeling
features = [col for col in df_train.columns if col not in ['date', 'demand', 'store', 'item']]
target = 'demand'

# split the data
X_train = df_train[df_train['date'] < split_date][features]
y_train = df_train[df_train['date'] < split_date][target]
X_valid = df_train[df_train['date'] >= split_date][features]
y_valid = df_train[df_train['date'] >= split_date][target]