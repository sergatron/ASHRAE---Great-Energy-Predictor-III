# Building Energy Usage Modeling
This is a capstone project for [Udacity's](https://www.udacity.com/) Data Science Nanodegree program and also is a Kaggle competition.

Competition [link](https://www.kaggle.com/c/ashrae-energy-prediction/overview)

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

# Table of Contents
- [**Exploratory Data Analysis (EDA)**](https://github.com/sergatron/ASHRAE---Great-Energy-Predictor-III/blob/master/EDA/eda.ipynb)
- [**ML model notebook**](https://github.com/sergatron/ASHRAE---Great-Energy-Predictor-III/blob/master/models/model_nb.ipynb)
- [**ML last model script**](https://github.com/sergatron/ASHRAE---Great-Energy-Predictor-III/blob/master/models/lgbm_model.py)


# Usage
For reproducibility, this repo may be cloned. Once cloned, scripts must be executed to train model. Then the web app may be run locally. 

In general, the following steps need to performed:
 1. Clone repo
 2. Setup virtual environment
 3. Train model
 4. Make predictions w/model
 5. Run web app


#### 1. Clone repo (git Bash):

```$ git clone https://github.com/sergatron/ASHRAE---Great-Energy-Predictor-III.git```


#### 2. Setup virtual envvironment w/Conda

NOTE: *requirements.yml* includes the name for virtual environment; this can easily be changed
```
conda env create -f requirement.yml
conda activate ashrae
```

#### 3. Train Model:

This will take some time to train and will output model to "models/lgbm_model.pkl" by default.
```python models/lgbm_model.py```


#### 4. Make Predictions

This will take some time and will output predictions to CSV file named "final_preds_df.csv"
```python models/submission_df.py```


#### 5. Run web app:

```python app/run.py```


# File Description
 - **app/**: scripts and html templates for the web app
 - **cleaning/**: Jupyter Notebook to test cleaning functions
 - **models/**: scripts to build and evaluate ML models. Also contains output *CSV* files of evaluation metrics.

# Acknowledgements
[Udacity](https://www.udacity.com/) Data Science Nanodegree 
[Kaggle](https://www.kaggle.com/c/ashrae-energy-prediction/overview) competition 

