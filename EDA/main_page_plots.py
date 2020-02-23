
import pandas as pd
import numpy as np

from plotly.graph_objs import Bar, Figure, Scatter, Histogram

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


#%%

def reduce_mem_usage(df, use_float16=False):
    """
    Original function code is from:
        https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08


    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024**2
    # print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

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
    # print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    # print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df

def load_data(filepath, **kwargs):
    """
    Import data from database into a DataFrame.

    Params:
    -------
        database_filepath: file path of database

    Returns:
    --------
        pd.DataFrame; (X, Y, df, category_names)
        Features and predictors, `X` and `Y`, respectively.
        Original DataFrame, `df`, and named of target labels
    """
    df = pd.read_csv(filepath, **kwargs)


    return df

#%%


# df = reduce_mem_usage(df)
# load predictions DataFrame
# preds_df = load_data('models/final_preds_df.csv')

#%%

def make_time_series(df, resample_by='D'):
    """
    Convert input DataFrame to time series.

    """
    dt_format = "%Y-%m-%d %H:%M:%S"
    df['timestamp'] = pd.to_datetime(df['timestamp'],
                                     format=dt_format)
    df.set_index('timestamp', inplace=True)

    # resample to daily average
    df = df.resample(resample_by).mean()

    return df


def get_figures():
    # load data
    df = load_data('data/clean_data.csv',
                   usecols=['meter', 'timestamp', 'meter_reading',
                          'site_id', 'primary_use','square_feet',
                          'year_built','air_temperature']
                   )

    # time-series prediction data
    df_ts = make_time_series(df)
    df_ts = reduce_mem_usage(df_ts)

    # define x and y
    ts_y = df_ts['meter_reading'].values
    ts_x = df_ts.index

    # Site ID counts
    site_counts = df['site_id'].value_counts().values
    site_id = df['site_id'].value_counts().index.to_list()

    # Building type counts
    bldg_type_counts = df['primary_use'].value_counts().values
    bldg_type = df['primary_use'].value_counts().index.to_list()

    # Building Type Area
    bldg_energy = df.groupby(['primary_use'])['meter_reading'].median()
    bldg_energy

    # Building Area (square feet)
    x_hist = df['square_feet'].values

    # create visuals
    graphs = [
        {
            'data': [Bar(
                x = site_id,
                y = site_counts)],
            'layout': {
                'title': 'Site ID Distribution',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Site ID"}
                },
            },
        {
            'data': [Bar(
                x = bldg_type,
                y = bldg_type_counts)],
            'layout': {
                'title': 'Building Type Distribution',
                'yaxis': {'title': 'Count',
                          'type': 'linear'},
                'xaxis': {'title': 'Category',
                          'tickangle': -30,}
                }
            },
        {
            'data': [Scatter(
                x = ts_x,
                y = ts_y)],
            'layout': {
                'title': 'Buildings\' Meter Reading',
                'yaxis': {'title': 'Energy Use (kWh)'},
                'xaxis': {'title': 'Date'},
                }
            },
        {
            'data': [Histogram(
                x = x_hist,
                nbinsx = 80,
                histnorm = 'probability',)],
            'layout': {
                'title': 'Building Area Distribution',
                'yaxis': {'title': 'Probability'},
                'xaxis': {'title': 'Area (Sq. Ft.)'},
                }
            },
        {
            'data': [Bar(
                y = bldg_energy.sort_values().index.tolist(),
                x = bldg_energy.sort_values().values,
                orientation = 'h',)],
            'layout': {
                'title': 'Energy Use',
                'yaxis': {'title': 'kWh'},
                'xaxis': {'title': 'Primary Use'},
                }
            },


        ]

    return graphs


# def plot_predictions(date_range=("2017-01-01", "2017-12-31"), df=preds_df):
#     """


#     Params:
#     --------
#         df : DataFrame
#             DataFrame to use for plotting.

#         date_range : tuple, optional
#             The default is ("2017-01-01", "2017-12-31") which the based on
#             the test DataFrame and the predictions are made for these dates.

#     Returns:
#     --------
#         fig : Plotly Figure


#     """
#     # time-series prediction data
#     df_new = make_time_series(df)
#     df_new = reduce_mem_usage(df)

#     # define x and y
#     y = df_new['meter_reading'].values
#     x = df_new.index

#     # define axis params to re-use
#     xy_axis = dict(
#         gridcolor='rgb(225, 225, 225)',
#         gridwidth=0.25,
#         linecolor='rgb(100, 100, 100)',
#         linewidth=2,
#         showticklabels=True,
#         color='black'
#     )
#     # update x-axis params
#     x_axis = xy_axis.copy()
#     x_axis.update(dict(
#         ticks='outside',
#         tickfont=dict(
#             family='Arial',
#             color='rgb(82, 82, 82)',))
#         )

#     # new figure
#     fig = Figure([Scatter(x=x,
#                           y=y,
#                           name='Predictions (Daily Meter Reading)',
#                           line=dict(color='royalblue',
#                                     width=2))
#                     ])

#     # Edit layout
#     fig.update_layout(title='Predictions',
#                       yaxis_title='kWh',
#                       plot_bgcolor='white',
#                       yaxis=xy_axis,
#                       xaxis=x_axis,

#                      )
#     # Use date string to set xaxis range
#     fig.update_layout(
#         xaxis_range=[date_range[0], date_range[1]],
#         )

#     return fig

