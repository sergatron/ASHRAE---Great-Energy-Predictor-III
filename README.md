# Building Energy Usage Modeling
This is a capstone project for [Udacity's](https://www.udacity.com/) Data Science Nanodegree program based on a [Kaggle competition](https://www.kaggle.com/c/ashrae-energy-prediction/overview).

This competition aims to build counterfactual models to predict buildings' energy usage. A successful model should scale well and minimize the Root Mean Squared Log Error. Counterfactual models are estimates of energy usage *before* any improvements are made within the building. This estimate is then compared with the actual energy usage *after* the improvements to calculate energy usage and confirm that the improvements are in fact working.

#### Web Application
This project takes results a step further to create a web application to present the predictions given some user inputs. Its usage is presented below. Further work involves deploying the model but for the moment it must be run locally. Instructions on how to use are provided below.


# Table of Contents
- [**Project Summary**](https://github.com/sergatron/ASHRAE---Great-Energy-Predictor-III/blob/master/Project_Overview.md)
- [**Exploratory Data Analysis (EDA)**](https://github.com/sergatron/ASHRAE---Great-Energy-Predictor-III/blob/master/EDA/eda.ipynb)
- [**ML-Initial Models (Jupyter Notebook)**](https://github.com/sergatron/ASHRAE---Great-Energy-Predictor-III/blob/master/models/model_nb.ipynb)
- [**ML-LightGBM Model (Script)**](https://github.com/sergatron/ASHRAE---Great-Energy-Predictor-III/blob/master/models/lgbm_model.py)


# Structure
```
ashrae_capstone
  |
  |___ app
  |       |
  |       |__ run.py
  |       |__ templates
  |       |           |__ go.html
  |       |           |__ master.html
  |       |__ __init__.py
  |       |__ run.py
  |
  |___ data
  |        |__ building_metadata.csv
  |        |__ clean_data.csv
  |        |__ test.csv
  |        |__ train.csv
  |        |__ weather_test.csv
  |        |__ weather_train.csv
  |
  |___ EDA
  |       |__ eda.ipynb
  |       |__ main_page_plots.py
  |       |__ predictions_plot.ipynb
  |
  |___ models
             |__ lgbm_model.py
             |__ xgboost_model.by
             |__ predictions_plot.ipynb
             |__ grid_search_lgbm.py
             |__ lgbm_kfold_metrics_df.csv
             |__ xgboost_kfold_metrics_df.csv
            
  ```


# Usage
For reproducibility, this repo may be cloned. Once cloned, scripts must be executed to train model. Then the web app may be run locally. 

In general, the following steps need to performed to run the app locally:
 1. Clone repo
 2. Setup virtual environment
 3. Train model
 4. Make predictions w/model
 5. Run web app

NOTE: pickled model and final predictions are included in this repo. Therefore, steps **3** and **4** can be skipped. 


#### 1. Clone repo (git Bash):

```
$ git clone https://github.com/sergatron/ASHRAE---Great-Energy-Predictor-III.git
```


#### 2. Setup virtual envvironment w/Conda

NOTE: *requirements.yml* includes the name for virtual environment; this can easily be changed but by default is set to 'ashrae'.
```
conda env create -f requirement.yml
conda activate ashrae
```

#### 3. Train Model:

This will take some time to train and will output model to "models/lgbm_model.pkl" by default.

```
python models/lgbm_model.py
```


#### 4. Make Predictions

This will take some time and will output predictions to CSV file named "final_preds_df.csv"

```
python models/submission_df.py
```


#### 5. Run web app:

To run locally, execute the script below and then open your browser. Enter IP 127.0.0.1:5000

```
python runapp.py
```


# File Description
 - **app/**: scripts and html templates for the web app
 - **data/**: data files, *csv* format
 - **cleaning/**: Jupyter Notebook to test cleaning functions
 - **EDA/**: Jupyter Notebook with Exploratory Data Analysis
 - **models/**: scripts to build and evaluate ML models. Also contains output *CSV* files of evaluation metrics.

# Acknowledgements
- [Udacity](https://www.udacity.com/) Data Science Nanodegree 
- [Kaggle](https://www.kaggle.com/c/ashrae-energy-prediction/overview) competition 

