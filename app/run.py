
import re
import json

import pandas as pd
import numpy as np

from plotly.graph_objs import Bar
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
df = load_data('../data/clean_data.csv')
df = reduce_mem_usage(df)

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


# # web page that handles user query and displays model results
# @app.route('/go')
# def go():
#     """
#     Renders a page which takes in user's query then passes
#     the query to the model which makes predictions and outputs
#     the labels to screen.
#     """
#     # save user input in query
#     query = request.args.get('query', '')

#     # use model to predict classification for query
#     classification_labels = model.predict([query])[0]
#     classification_results = dict(zip(df.columns[4:], classification_labels))

#     # This will render the go.html Please see that file.
#     return render_template(
#         'go.html',
#         query=query,
#         classification_result=classification_results
#     )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()