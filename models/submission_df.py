
import pandas as pd
import numpy as np

from joblib import load

import datetime
import gc

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype



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

def get_nan_idx(column, df):
    """
    Returns the indices of missing values in given
    column and DataFrame.
    """
    return df[df[column].isna()].index.values

def get_nan_columns(df):
    """
    Extract name of columns which have missing values.
    """
    df = nan_val_summary(df)
    return df[df['fraction_missing'] > 0]['columns'].values




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
            Name of column to fill

    Returns:
    --------
        None
        DataFrame is updated inplace.
    """
    # aggregate data to obtain median value for a particular site, month, and day
    agg_weather_df = pd.DataFrame(df.groupby(['site_id', 'month', 'day'])[column].agg(agg_func))

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

def fill_nans(df):
    """
    Fill missing values
    """
    nan_cols = get_nan_columns(df)
    [fill_weather_nans(col, df) for col in nan_cols]

def add_features(df):

    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.weekday
    df['log_square_ft'] = np.log1p(df['square_feet'])
    # df['air_temp_cos'] = np.cos(df['air_temperature'])
    # df['air_temp_sin'] = np.sin(df['air_temperature'])


# ## Load Test Data
print("Loading data...\n")
# Load data
# train_df = pd.read_csv('../data/train.csv')
building_df = pd.read_csv('../data/building_metadata.csv')

test_df = pd.read_csv('../data/test.csv')
weather_test_df = pd.read_csv('../data/weather_test.csv')



# Convert to Datetime
dt_format = "%Y-%m-%d %H:%M:%S"
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'],
                                       format=dt_format)
weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'],
                                         format=dt_format)

print('Merging data... \n')
# MERGE DATA
tb_df = pd.merge(test_df, building_df, on='building_id')
test_data_df = pd.merge(tb_df, weather_test_df, on=['site_id', 'timestamp'])
del test_df, building_df, weather_test_df, tb_df
gc.collect()

print('Test data shape: \n')
print(test_data_df.shape)


test_data_df = reduce_mem_usage(test_data_df)


# ## Load Model
print('Loading model... \n')
model = load('lgbm_model.pkl')


print("Feature importance:\n")
sorted(list(zip(model.feature_name(),
                model.feature_importance())), key=lambda x: x[1], reverse=True)


# ## Prepare Data
# Clean and process testing data using the same functions as used during training.
print("--- Processing data for predictions ---\n")

print('Adding features...\n ')
add_features(test_data_df)
print('Filling missing values... \n')
fill_nans(test_data_df)

# drop features
print('Dropping features... \n')
test_timestamp = test_data_df['timestamp'].values

drop_cols = ['timestamp', 'year_built', 'floor_count',
              'precip_depth_1_hr', 'wind_speed', 'wind_direction',
              'cloud_coverage',
              'building_id',
              ]

test_data_df.drop(drop_cols, axis=1, inplace=True)

# extract variables to keep
# row_ids for predictions, then drop row_id
row_ids = test_data_df['row_id']
site_ids = test_data_df['site_id']
test_data_df.drop('row_id', axis=1, inplace=True)


# ## Make Predictions
# Once the date is ready, make predictions on the test data.

# NOTE: use np.expm1 to convert back to kWh since the model was trained
# to predict a transformed target

print("Making predictions... \n")
final_predictions = np.expm1(model.predict(test_data_df))
test_data_df.shape[0] == final_predictions.shape[0]

# check for negative values
print('Negative predictions:', (final_predictions < 0).any())


print('Best score achieved by model:\n', model.best_score)



# ## Save Predictions Array
# print('Saving predictions array... \n')
# np.save('test_predictions', final_predictions)


# ## Make Submission DataFrame
# clip predictions: if pred is negative, set it to zero
final_preds_df = pd.DataFrame({"row_id": row_ids,
                               "meter_reading": np.clip(final_predictions, 0, a_max=None),
                               'site_id': site_ids})
del row_ids, final_predictions
gc.collect()

# add timestamp
final_preds_df['timestamp'] = test_timestamp

print('Saving predictions DataFrame... \n')
final_preds_df.to_csv('final_preds_df.csv', encoding='utf-8', index=False)

print("All Done!")
print('-'*75)
