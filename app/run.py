
import re
import json

import pandas as pd
import numpy as np

from plotly.graph_objs import Bar, Figure, Scatter, Histogram
import plotly

from flask import Flask
from flask import render_template, request, jsonify

from joblib import dump, load
from sqlalchemy import create_engine


from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


app = Flask(__name__)

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
    # print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    # print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df

def load_data(filepath):
    """
    Import data from database into a DataFrame. Split DataFrame into
    features and predictors, `X` and `Y`.

    Preprocess data.

    Params:
    -------
        database_filepath: file path of database

    Returns:
    --------
        pd.DataFrame; (X, Y, df, category_names)
        Features and predictors, `X` and `Y`, respectively.
        Original DataFrame, `df`, and named of target labels
    """
    df = pd.read_csv(filepath)


    return df

#%%

# load model
# model = load("models/lgbm_model.pkl")

# load data
df = load_data('data/clean_data.csv')
df = reduce_mem_usage(df)



#%%

# plot predictions
def make_sample_df(df, date_range=('2017-01', '2017-03')):
    """
    Create new sample DataFrame with sample meter readings, and new
    timestamp given `date_range`.
    """
    df_new = pd.DataFrame()
    df_new['timestamp'] = np.arange(date_range[0], date_range[1], dtype='datetime64[D]')

    # get number of samples
    n_samples = df_new.shape[0]

    # sample from original DF
    df.reset_index(drop=True, inplace=True)
    df_new['meter_reading'] = df['meter_reading'].sample(n_samples).values
    # set new index to `timestamp`
    df_new = df_new.set_index('timestamp')

    return df_new

def make_sample_plot(df, date_range=('2017-01', '2017-03'), resample='D'):
    df_new = make_sample_df(df, date_range)
    df_new = df_new.resample(resample).mean()
    df_new['meter_reading'].plot(
        kind='line',
        label='TS Plot',
        legend=True,
        linewidth=2
    )

def plot_predictions(df):

    # get plot for predictions
    df_new = make_sample_df(df)

    # define two lines to plot
    y = df_new['meter_reading'].values
    y2 = np.cos(df_new['meter_reading'].values) + 29

    # new figure
    fig = Figure([Scatter(x=df_new.index,
                                y=df_new['meter_reading'].values,
                                name='Predictions',
                                line=dict(color='blue'),)
                    ])
    # add another line
    fig.add_trace(Scatter(x=df_new.index,
                             y=y2,
                             name='Sine',
                             line=dict(
                                 color='firebrick',
                                 width=4,
                                 dash='dot')
                             )
                  )
    # Edit layout
    fig.update_layout(title='Predictions',
                      yaxis_title='kWh',
                      plot_bgcolor='white',

                      yaxis=dict(
                          showgrid=True,
                          zeroline=True,
                          showline=True,
                          linecolor='rgb(204, 204, 204)',
                          showticklabels=True,
                          color='black'
                      ),
                      xaxis=dict(
                          showline=True,
                          showgrid=True,
                          showticklabels=True,
                          linecolor='rgb(204, 204, 204)',
                          linewidth=2,
                          ticks='outside',
                          tickfont=dict(
                              family='Arial',
                              size=12,
                              color='rgb(82, 82, 82)',
                          ),
                      )
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


    # extract data needed for visuals
    site_counts = df['site_id'].value_counts().values
    site_id = df['site_id'].value_counts().index.to_list()

    # extract data needed for visuals
    bldg_type_counts = df['primary_use'].value_counts().values
    bldg_type = df['primary_use'].value_counts().index.to_list()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(x = site_id,
                    y = site_counts)
                ],

            'layout': {
                'title': 'Site ID Distribution',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Site ID"}
                }
            },
        {
            'data': [
                Bar(x = bldg_type,
                    y = bldg_type_counts)
                ],
            'layout': {
                'title': 'Building Type Distribution',
                'yaxis': {'title': 'Count',
                          'type': 'linear'
                          },
                'xaxis': {'title': 'Category',
                          'tickangle': -45,
                          }
                }
            },
        ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Renders a page which takes in user's query then passes
    the query to the model which makes predictions and outputs
    the labels to screen.
    """
    # save user input in query
    query_1 = request.args.get('query-1', 'NA')
    query_2 = request.args.get('query-2', 'NA')
    query_3 = request.args.get('query-3', '')
    query_4 = request.args.get('query-4', '')


    # get predictions plot
    fig = plot_predictions(df)

    # encode plotly graphs in JSON
    graphJSON = fig.to_json()

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query_1=query_1,
        query_2=query_2,
        query_3=query_3,
        query_4=query_4,
        graphJSON=graphJSON
        # ids=ids,
        # graphJSON=graphJSON
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()