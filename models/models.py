
#%%
import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import datetime
import gc

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
#%%
# LOAD DATA
def load_data(train_path='../data/train.csv',
              building_path='../data/building_metadata.csv',
              weather_path='../data/weather_train.csv'):
    """

    Params:
    --------
        train_path : str
            '../data/train.csv'

        building_path : str
            '../data/building_metadata.csv'

        weather_path : str
            '../data/weather_train.csv'

    Returns:
    --------
        Tuple of DataFrames

    """

    train_df = pd.read_csv(train_path)
    building_df = pd.read_csv(building_path)
    weather_df = pd.read_csv(weather_path)

    return train_df, building_df, weather_df


#%%

def reduce_mem_usage(df, use_float16=False):
    """
    Original function code is from:
        https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08


    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def to_datetime(df):
    dt_format = "%Y-%m-%d %H:%M:%S"
    df['timestamp'] = pd.to_datetime(df['timestamp'],format=dt_format)
    return df

def nan_val_summary(df):
    """Summarize the counts of missing values"""
    nan_arr = np.count_nonzero(df.isnull(), axis=0)
    frac = nan_arr / df.shape[0]
    nan_df = pd.DataFrame(
        {'columns': df.columns,
         'nan_count': nan_arr,
         'fraction_missing': frac}
                 )
    return nan_df

def add_features(df):

    # time features
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['weekend'] = df['timestamp'].dt.weekday()

    # energy features
    df['kwh_per_sqft'] = df['meter_reading'] / df['square_feet']
    df['kwh_per_month'] = df['meter_reading'] / df['month']
    return df


def fill_weather_nans(column, df, agg_func='median'):
    """
    Fills in missing values in Weather data. Column name must be provided.
    Function fills missing values inplace and therefore returns nothing.

    Uses the following method to fill NaNs:
        1. Groupby three variables, `site_id`, `month`, and `day`
        2. Compute `agg_func`, this will be used to fill missing values
        3. Check for missing values in aggregated data
            - If missing values present
                - Use interpolation to fill those missing values
        4. Set index of DataFrame `df`  such that it matches the aggregated
        DataFrame index
        5. Update DataFrame `df` and fill missing values
        6. Reset index of `df`

    Params:
    -------
        column: str
            Name of column to fill.

        df: DataFrame
            DataFrame to fill missing values.

        agg_func: str
            Aggregate function to utilize on grouped data.

    Returns:
    --------
        None
        DataFrame is updated inplace.
    """
    # convert to datetime
    df = to_datetime(df)

    # new features: month, hour
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day

    # aggregate data to obtain median value for a particular
    # site, month, and day
    agg_weather_df = pd.DataFrame(df.groupby(['site_id', 'month', 'day'])\
                                  [column].agg(agg_func))

    # check for missing values in the aggregated data
    if agg_weather_df[column].isnull().any():
        # fill NaNs using interpolation
        agg_df = agg_weather_df[column].interpolate(limit_direction='both',
                                                    inplace=True)
        agg_weather_df.update(agg_df, overwrite=False)

    # set index before updating input DataFrame
    df.set_index(['site_id', 'month', 'day'], inplace=True)
    df.update(agg_weather_df, overwrite=False)

    # reset index
    df.reset_index(inplace=True)


def get_nan_columns(df):
    """
    Extract name of columns which have missing values.
    """
    df = nan_val_summary(df)
    return df[df['fraction_missing'] > 0]['columns'].values



def clean_data(drop=False):

    # LOAD DATA
    print('Loading data...\n')
    train_df, building_df, weather_df = load_data()
    # convert to datetime object
    train_df = to_datetime(train_df)

    # CONVERSION AT SITE 0
    # TODO

    # FILL MISSING WEATHER DATA
    # get columns with missing values, then call function on columns
    print('Filling missing values...\n')
    nan_cols = get_nan_columns(weather_df)
    [fill_weather_nans(col, weather_df) for col in nan_cols]

    # MERGE DATA
    print('Merging data...\n')
    tb_df = pd.merge(train_df, building_df, on='building_id')
    df = pd.merge(tb_df, weather_df, on=['site_id', 'timestamp'])

    # REDUCE MEM USAGE
    print('Reducing memory usage...\n')
    df = reduce_mem_usage(df, use_float16=True)
    print(df.info())
    print('\n')

    # drop data
    if drop:
        print('Dropping building 1099...\n')
        # drop indices with building 1099
        idx = np.where(df['building_id'] == 1099)[0]
        # df.loc[idx]
        df_no1099 = df.drop(idx, axis=0)
        return df_no1099

    return df


#%%

if __name__ == '__main__':
    df = clean_data()

    print(df.head())
