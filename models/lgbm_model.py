
#%%
import pandas as pd
import numpy as np
import pickle
from joblib import dump, load
import lightgbm as lgbm
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import make_scorer

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.decomposition import PCA

import datetime
import gc

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

#%%

# Load data
train_df = pd.read_csv('../data/train.csv')
building_df = pd.read_csv('../data/building_metadata.csv')
weather_df = pd.read_csv('../data/weather_train.csv')


# In[59]:


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


# # Convert to Datetime

# In[60]:


dt_format = "%Y-%m-%d %H:%M:%S"
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], format=dt_format)
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], format=dt_format)


# # Merge Data

# In[61]:


tb_df = pd.merge(train_df, building_df, on='building_id')
all_df = pd.merge(tb_df, weather_df, on=['site_id', 'timestamp'])
del train_df, building_df, weather_df, tb_df


# In[63]:


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
    df['log_square_ft'] = np.log1p(df['square_feet'])

def get_sample(df, n=0.5):
    n_sample = np.int32(df.shape[0] * n)
    return df.sample(n_sample)


# ## Drop Building 1099

# In[64]:


# drop indices with building 1099

# idx = np.where(all_df['building_id'] == 1099)[0]
# # all_df.loc[idx]
# df_no1099 = all_df.drop(idx, axis=0)


# In[65]:
def save_model(model, filepath):
    """

    Pickles model to given file path.

    Params:
    -------
        model: Pipeline
            Model to pickle.

        filepath: str
            save model to this directory

    Returns:
    -------
        None.

    """
    try:
        dump(model, filepath)
    except Exception as e:
        print(e)
        print('Failed to pickle model.')

def get_X_y(df=all_df, n=1.0):

    if n < 1.0:
        # take sample from data
        df = get_sample(df, n)

    # reset index
    df.reset_index(drop=True, inplace=True)

    # add features
    add_features(df)


    # drop features
    drop_cols = ['timestamp', 'year_built', 'floor_count']
    df.drop(drop_cols, axis=1, inplace=True)

    # reduce memory usage
    df = reduce_mem_usage(df)

    # define feature space X, and target y
    X = df.drop('meter_reading', axis=1)
    y = df['meter_reading']


    del df

    gc.collect()

    return X, y

#%%
# ## LightGBM
# define metric, Root Mean Squared Log Error
def RMSLE(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

# grid search function
def perform_grid_search(model, params, X_train, y_train, drop_feats=False, **kwargs):

    if drop_feats:
        # drop categorical column containing strings to speed up training
        if X_train.columns.isin(['primary_use']).any():
            X_train = X_train.drop(['primary_use'], axis=1)
            # X_test = X_test.drop(['primary_use'], axis=1)

    # , greater_is_better=False --> -1.6236
    gs_cv = GridSearchCV(
        model,
        params,
        # scoring=make_scorer(RMSLE),
        cv=3,
        n_jobs=-1,
        **kwargs
        )
    print('Searching for optimal params...\n')
    gs_cv.fit(X_train, y_train)
    print(gs_cv.best_params_)

    return gs_cv

#%%

# define feature space and target
X, y = get_X_y(n=0.1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

print("Dropping features: [primary_use]")
X.drop(['primary_use'], axis=1, inplace=True)

print('X-shape:', X.shape)
print('\n')



#%%


# create splits
kf = KFold(n_splits=5)

categorical_features = ["building_id",
                        "site_id",
                        "meter",
                        # "primary_use",
                        ]
params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 1800,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": "rmse",
    }

models = []
for train_idx, test_idx in kf.split(X):

    # define train data
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    print('X train shape:', X_train.shape)
    print('y train shape:', y_train.shape)

    # define test data
    X_test = X.loc[test_idx]
    y_test= y.loc[test_idx]

    # fill missing data
    print('Filling missing training data...\n')
    fill_nans(X_train)

    # fill missing data
    print('Filling missing test data...\n')
    fill_nans(X_test)

    # lgbm train data
    d_training = lgbm.Dataset(X_train,
                             label=y_train,
                             categorical_feature=categorical_features,
                             free_raw_data=False
                             )
    # lgbm test data
    d_test = lgbm.Dataset(X_test,
                         label=y_test,
                         categorical_feature=categorical_features,
                         free_raw_data=False
                         )

    # train model
    print('Training model...\n')
    model = lgbm.train(params,
                      train_set=d_training,
                      num_boost_round=1000,
                      valid_sets=[d_training, d_test],
                      verbose_eval=100,
                      early_stopping_rounds=50)

    # make predictions
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # score predictions
    print('Test RMSLE:', RMSLE(y_test, y_pred))
    print('Test RMSE:', MSE(y_test, y_pred, squared=False))
    print('Test MAE:', MAE(y_test, y_pred))
    print('=' * 75)
    print('\n')

    # models.append(model)
    print('=' * 75)

