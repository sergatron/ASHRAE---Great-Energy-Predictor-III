import itertools
import time
import datetime
import gc
import time
import pickle
from joblib import dump, load

import pandas as pd
import numpy as np
import lightgbm as lgbm
import xgboost as xgb

import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


plt.style.use('seaborn-white')
rcParams['axes.labelsize'] = 'x-large'
rcParams['axes.edgecolor'] = 'black'
rcParams['axes.facecolor'] = 'white'
rcParams['axes.titlesize'] = 'x-large'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.xmargin'] = 0.02
rcParams['axes.ymargin'] = 0.02

rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = ':'
rcParams['grid.alpha'] = 0.2
rcParams['grid.color'] = 'black'

rcParams['figure.titlesize'] = 'x-large'
rcParams['figure.edgecolor']= 'black'
rcParams['figure.facecolor'] = 'white'
rcParams['figure.figsize'] = [12, 8]

rcParams['ytick.labelsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'

# format float in pandas
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.max_columns = 30
pd.options.display.max_rows = 50
pd.options.display.width = 100



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

def get_sample(df, n=0.5):
    n_sample = np.int32(df.shape[0] * n)
    return df.sample(n_sample)


def preprocess_data(df, quantile=0.6):
    # drop indices with building 1099
    print('Removing noisy data... \n')
    # get indices of building ID 1099
    idx = np.where(df['building_id'] == 1099)[0]
    # drop specified indices
    df = df.drop(idx, axis=0)

    # filter outliers
    # discard heavy use, and low energy usage
    q = df['meter_reading'].quantile(quantile)
    df = df[(df['meter_reading'] < q) & \
            (df['meter_reading'] >= 1.0)]

    return df

def get_X_y(df, n=1.0, quantile=0.6, filter_meter=True):

    # filter outliers
    # discard heavy use, and low energy usage
    if filter_meter:
        df = preprocess_data(df, quantile=quantile)

    # sample size
    if n < 1.0:
        print(f'Working with {n} sample fraction of data\n')
        # take sample from data
        df = get_sample(df, n)

    # reset index
    df.reset_index(drop=True, inplace=True)

    # add features
    add_features(df)


    # drop features
    # NOTE: drop `primary_use` for XGBoost
    drop_cols = ['timestamp', 'year_built', 'floor_count',
                 'precip_depth_1_hr', 'wind_speed', 'wind_direction',
                 'cloud_coverage', 'primary_use']

    df.drop(drop_cols, axis=1, inplace=True)

    # reduce memory usage
    df = reduce_mem_usage(df)

    # define feature space X, and target y
    X = df.drop('meter_reading', axis=1)
    y = df['meter_reading']


    del df

    gc.collect()

    return X, y

# define metric, Root Mean Squared Log Error
def RMSLE(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

def show_metrics(y_true, y_pred):
    # print params and metric
    print('-' * 75)
    print('RMSLE:', np.round(RMSLE(y_true, y_pred), 4))
    print('RMSE: ', np.round(MSE(y_true, y_pred, squared=False), 4))
    print('MAE:  ', np.round(MAE(y_true, y_pred), 4))
    print('-' * 75)

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


def train_lgbm_model(X_train, y_train, X_test, y_test,
                params, boost_rounds=1000, plot=False):


    categorical_features = ["building_id",
                            "site_id",
                            "meter",
                            "primary_use",
                            ]
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
    model = lgbm.train(params,
                      train_set=d_training,
                      num_boost_round=boost_rounds,
                      valid_sets=[d_training, d_test],
                      verbose_eval=100,
                      early_stopping_rounds=50)
    if plot:
        # plot feature importance
        lgbm.plot_importance(model);

    del X_train, X_test, y_train, y_test, d_training, d_test
    gc.collect()

    return model

def train_xgboost_model(X_train, y_train, X_test, y_test, params,
                        boost_rounds=500, plot=False):

    # xgboost train data
    d_training = xgb.DMatrix(X_train, label=y_train)
    # xgboost test data
    d_test = xgb.DMatrix(X_test, label=y_test)

    # define evaluation method
    evallist = [(d_training, 'train'), (d_test, 'eval')]

    print('\nTraining XGBoost model... \n')
    # train model
    model = xgb.train(params,
                      dtrain=d_training,
                      num_boost_round=boost_rounds,
                      evals=evallist,
                      early_stopping_rounds=50,
                      verbose_eval=100,

                      )
    if plot:
        # plot feature importance
        xgb.plot_importance(model);

    del X_train, X_test, y_train, y_test, d_training, d_test
    gc.collect()

    return model


def train_cv_model(X, y, params, boost_rounds=500, scale=False, n_splits=4):

    # pipeline to transform data
    scaler = StandardScaler()
    # pca = PCA(n_components=3, random_state=11)
    transfomer = make_pipeline(
        scaler,
        # pca
        )

    # create splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=11)
    # models = []

    # instantiate metric arrays
    train_rmsle = np.array([])
    test_rmsle = np.array([])
    # predictions = np.array([])
    kfold = 0
    for train_idx, test_idx in kf.split(X):
        print('\n')
        print('-'*75)
        print('\nEvaluating Fold:', kfold)

        # define train data
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]

        # define test data
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        print('Filling missing data...\n')
        fill_nans(X_train), fill_nans(X_test)

        if scale:
            print('Scaling data... \n')
            X_train = transfomer.fit_transform(X_train)
            X_test = transfomer.fit_transform(X_test)

        # xgboost train data
        d_training = xgb.DMatrix(X_train, label=y_train)
        # xgboost test data
        d_test = xgb.DMatrix(X_test, label=y_test)

        # define evaluation method
        evallist = [(d_training, 'train'), (d_test, 'eval')]

        print('\nTraining XGBoost model... \n')
        # train model
        model = xgb.train(params,
                          dtrain=d_training,
                          num_boost_round=boost_rounds,
                          evals=evallist,
                          early_stopping_rounds=50,
                          verbose_eval=100,

                          )

        print('Evaluating predictions... \n')
        # make predictions
        y_pred = (model.predict(d_test))
        y_pred_train = (model.predict(d_training))

        # print metric
        print('\nTest Metrics:')
        show_metrics(y_test, y_pred)
        # print metric
        print('\nTrain Metrics:')
        show_metrics(y_train, y_pred_train)

        # score train and test sets
        rmsle_test_score = RMSLE(y_test, y_pred)
        rmsle_train_score = RMSLE(y_train, y_pred_train)
        # append scores
        train_rmsle = np.append(train_rmsle, rmsle_train_score)
        test_rmsle = np.append(test_rmsle, rmsle_test_score)

        del X_train, X_test, y_train, y_test, model, y_pred, y_pred_train
        gc.collect()

        kfold += 1

    print('\nMean Train RMSLE:', np.mean(train_rmsle))
    print('Mean Test RMSLE:', np.mean(test_rmsle))
    print('='*75)

    # save metrics
    metrics_df = pd.DataFrame()
    metrics_df['train_rmsle'] = train_rmsle
    metrics_df['test_rmsle'] = test_rmsle
    metrics_df['num_folds'] = [n_splits] * train_rmsle.shape[0]
    metrics_df.to_csv('xgb_kfold_metrics_df.csv', encoding='utf-8', index=False)
    return metrics_df

def load_data():
    # LOAD DATA
    print("Loading data...\n")
    train_df = pd.read_csv('../data/train.csv')
    building_df = pd.read_csv('../data/building_metadata.csv')
    weather_df = pd.read_csv('../data/weather_train.csv')
    return train_df, building_df, weather_df

def to_datetime(df, column, dt_format = "%Y-%m-%d %H:%M:%S"):
    # Convert to Datetime
    return pd.to_datetime(df[column], format=dt_format)



def main(params, boost_rounds=500, data_sample=1.0, output_model=False,
         out_arr_path='predictions.npy', output_model_path='xgb_model.pkl',
         validate=False, quantile=0.5, scale=False, cv_splits=3):

    # LOAD DATA
    train_df, building_df, weather_df = load_data()

    # Convert to Datetime
    weather_df['timestamp'] = to_datetime(weather_df, 'timestamp')
    train_df['timestamp'] = to_datetime(train_df, 'timestamp')

    # MERGE DATA
    tb_df = pd.merge(train_df, building_df, on='building_id')
    all_df = pd.merge(tb_df, weather_df, on=['site_id', 'timestamp'])
    del train_df, building_df, weather_df, tb_df

    # PROCESS DATA
    # define x, and y
    X, y = get_X_y(df=all_df,
                   n=data_sample,
                   quantile=quantile,
                   filter_meter=True
                   )

    # TRAIN/VALIDATE
    # cross validate model with given params
    if validate:
        print('Performing cross-validation... \n')
        validation_metrics_df = train_cv_model(
            X,
            y,
            params=params,
            boost_rounds=boost_rounds,
            scale=scale,
            n_splits=cv_splits)

        print('Saving metrics and parameters... \n')

        validation_metrics_df.to_csv("xgb_kfold_metrics_df.csv", index=False)


    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



    print('Filling missing data...\n')
    fill_nans(X_train), fill_nans(X_test)

    # print("Dropping features: [primary_use]")
    # X.drop(['primary_use'], axis=1, inplace=True)

    print('X-shape:', X.shape)
    print('\n')

    print('Training model...\n')
    # xgboost train data
    d_training = xgb.DMatrix(X_train, label=y_train)
    # xgboost test data
    d_test = xgb.DMatrix(X_test, label=y_test)

    # define evaluation method
    evallist = [(d_training, 'train'), (d_test, 'eval')]

    print('\nTraining XGBoost model... \n')
    # train model
    model = xgb.train(params,
                      dtrain=d_training,
                      num_boost_round=boost_rounds,
                      evals=evallist,
                      early_stopping_rounds=50,
                      verbose_eval=100,

                      )

    # make predictions
    print('Evaluating predictions... \n')
    y_pred = (model.predict(d_test))
    y_pred_train = (model.predict(d_training))

    print('\nTest Metrics:')
    show_metrics(y_test, y_pred)
    print('\nTrain Metrics:')
    show_metrics(y_train, y_pred_train)

    if output_model:
        # save model to file
        print('Saving model... \n')
        save_model(model, output_model_path)

        # # save predictions to file
        # print('Saving predictions array... \n')
        # np.save(out_arr_path, y_pred_train)
        # print('Saving complete!')


    print('All done!')
    return RMSLE(y_test, y_pred)


if __name__ == '__main__':

    xgb_params = dict(
        obj='regression',
        feval=RMSLE,
        tree_method='hist',
        n_estimators=80,
        max_depth=10,
        min_child_weight=3,
        eta=0.3,
        subsample=0.7,
        gamma=4,

        )

    start_time = time.perf_counter()

    # execute main function
    score = main(
        params=xgb_params,
        boost_rounds=200,
        data_sample=1.0,
        output_model=False,
        output_model_path='xgb_model_.pkl',
        quantile=0.7,
        scale=False,
        out_arr_path='xgb_train_predictions.npy',
        validate=False,
        cv_splits=3
        )
    end_time = time.perf_counter()

    print('\nTime elapsed:', np.round((end_time - start_time)/60, 4), 'minutes.')
    print('-'*75)

    # # HYPER-PARAM SEARCH
    # # print('='*75)
    # print('Performing Hyper-param search... ')
    # print('='*75)
    # # Hyperparameter grids
    # input_param = {'num_leaves': [1800, 2100],
    #                 'n_estimators': [60, 80],
    #                 "reg_lambda": [2, 4],

    #                 }
    # results = {}


    # # For each couple in the grid
    # for var1, var2, var3 in itertools.product(input_param['num_leaves'],
    #                                     input_param['reg_lambda'],
    #                                     input_param['n_estimators']):
    #     params['num_leaves'] = var1
    #     params['n_estimators'] = var3
    #     params['reg_lambda'] = var2
    #     print('\n')
    #     print(params)
    #     # execute model training
    #     score = main(
    #         params=params,
    #         boost_rounds=800,
    #         data_sample=1.0,
    #         output_model=False,
    #         output_model_path='lgbm_model.pkl',
    #         quantile=0.7,
    #         scale=False,
    #         out_arr_path='train_predictions.npy',
    #         validate=False
    #         )

    # results[(var1, var2, var3)] = score
    # # sort output, smallest value is best
    # best_params = sorted(results.items(),
    #                       key=lambda x: x[1],
    #                       reverse=False)[0][0]

    # print('\nBest params:', best_params)
    #     # TODO: record metrics for each param set;
    #     #       pick params with best metric for Test Set

