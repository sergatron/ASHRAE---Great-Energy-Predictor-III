from app import app
import re
import json

import pandas as pd
import numpy as np

from plotly.graph_objs import Bar, Figure, Scatter, Histogram
import plotly

from flask import Flask
from flask import render_template, request, jsonify

from joblib import dump, load

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

from EDA.main_page_plots import get_figures


pd.options.display.float_format = '{:.4f}'.format


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


def make_time_series(df, resample_by='D'):
    """
    Convert input DataFrame to time series.

    """
    dt_format = "%Y-%m-%d %H:%M:%S"
    df['timestamp'] = pd.to_datetime(df['timestamp'],
                                      format=dt_format)
    df = df.set_index('timestamp')

    # resample to daily average
    df = df.resample(resample_by).mean()
    df = reduce_mem_usage(df)

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

# load predictions DataFrame
preds_df = load_data('models/final_preds_df.csv',
                     usecols=['meter_reading', 'timestamp'])

def plot_predictions(date_range=("2017-01-01", "2017-12-31"), df=preds_df):

    """


    Params:
    --------
        df : DataFrame
            DataFrame to use for plotting.

        date_range : tuple, optional
            The default is ("2017-01-01", "2017-12-31") which the based on
            the test DataFrame and the predictions are made for these dates.

    Returns:
    --------
        fig : Plotly Figure


    """
    # # get plot for predictions
    # df_new = make_sample_df(df)

    # time-series prediction data
    df_new = make_time_series(df)

    # define x and y
    y = df_new['meter_reading'].values
    x = df_new.index

    # define axis params to re-use
    xy_axis = dict(
        gridcolor='rgb(225, 225, 225)',
        gridwidth=0.25,
        linecolor='rgb(100, 100, 100)',
        linewidth=2,
        showticklabels=True,
        color='black'
    )
    # update x-axis params
    x_axis = xy_axis.copy()
    x_axis.update(dict(
        ticks='outside',
        tickfont=dict(
            family='Arial',
            color='rgb(82, 82, 82)',))
        )

    # new figure
    fig = Figure([Scatter(x=x,
                          y=y,
                          name='Predictions (Daily Meter Reading)',
                          line=dict(color='royalblue',
                                    width=2))
                    ])

    # Edit layout
    fig.update_layout(title='Predictions',
                      yaxis_title='kWh',
                      plot_bgcolor='white',
                      yaxis=xy_axis,
                      xaxis=x_axis,
                      )
    # Use date string to set xaxis range
    fig.update_layout(
        xaxis_range=[date_range[0], date_range[1]],
        )

    return fig


#%%
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Renders graphs created from the loaded data.
    Renders homepage with visualizations of the data.

    """

    graphs = get_figures()
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """
    Renders a page displays the predictions for given date range.
    """
    # save user input in query
    start_date = request.args.get('start_date', 'NA')
    end_date = request.args.get('end_date', 'NA')


    # get predictions plot
    fig = plot_predictions(date_range=(start_date, end_date))

    # encode plotly graphs in JSON
    graphJSON = fig.to_json()

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        start_date=start_date,
        end_date=end_date,
        graphJSON=graphJSON
    )
