# Predicting Building's Energy Use

## Project Definition
### Overview

Mainting perfect indoor temperature in a skyscraper requires an extraordinary amount of energy which translates to money being spent to maintain ideal conditions. Additionally, this energy expenditure may have negative impact on the environment. Fortunately, investments are being made to reduce cost and emissions. Buildings which install new and/or upgrade their existing equiptment to more efficient ones can reduce energy consumption, cost, and environmental impact. However, how can these savings be measured? 


"Under pay-for-performance financing, the building owner makes payments based on the difference between their real energy consumption and what they would have used without any retrofits. The latter values have to come from a model." [1]

Additionally, how do we evaluate energy savings of the improvements? To do this, we need to estimate or model the amount of energy a building would have used *before* the improvements. After the improvements are made, we can compare the energy usage between the original building (modeled energy usage) and the retrofit building (actual energy usage). Then, the energy savings can be calculated due to the retrofit. 

The provided data was collected over a three year period from over 1,000 buildings across the world. To capture true savings, the model must be quite robust and accurate. The ideal model will be able to find patterns which contribute to energy consumption. It must scale well since the training data includes over 20 million samples. 

### Problem Statement
The goal is to create a model which can predict a building's energy use with minimal error. The first step is to use exploratory data analysis (EDA) to understand the nature of all present variables (continuous and discrete) through the use of statistical plots, descriptive and inferential statistics. This exploration will help locate missing values, outliers, and relationships between variables. Ideally, the knowledge gained in this section will be leveraged to build a machine learning model that minimizes prediction error. Removing noisy data is important no matter which ML algorithm is used. Two potential algorithms will be evaluated, LightBGM and XGBoost. Both are tree-based algorithms and should perform well on the large data set. 



### Metrics
The model will aim to minimize the Root Mean Squared Log Error. It is defined as 

#### $\sqrt{1/n \sum_{i=1}^{n} t_i (log(Pi+1)−log(Ai+1))^2}$


where $Pi$ are the predicted values and $Ai$ are the Actual values. Logarithmic properties lets us rewrite this as

#### $ log\frac {Pi + 1} {Ai + 1} $. 


## Analysis
### Data Exploration and Visualization
Please refer to Jupyter Notebook LINK for analysis and visualizations.


## Methodology
### Data Preprocessing
There are three data sets provided for training in the form of a *csv* file, `train.csv`, `weather_train.csv`, and `building_metadata.csv`. All three files were inspected for missing values before merging together. Missing weather data was filled using aggregated data based on the location (site_id), month, and day of the missing value. Incorporating these variables insured a relatively accruate filling method as ooposed to simply using median or mean of entire data to fill missing values. Building metadata also contained missing values in `floor_count` and `year_built` variables. However, `floor_count` seemed to be an uninformative variable compared to `square_feet` and therefore was removed. The total area of a building is a more infromative variable than the amount of floors a building contains. The `year_built` variable contained too many missing values (over 60%) and was removed. 

In order to minimize model's error, removing noisy samples and features was of utmost importance. Even though LightGBM is a tree-based model, it is still important to remove noisy data and try to isolate the true signal. In particular, some buildings were found to have an enormous energy consumption as compared to the average consumption. Most likely, these buildings are relatively old with poor insulation and an inefficient steam heating system. These samples are not representative of the population and are therefore exluded from the training data set. 

### Implementation
Two models were built and evaluated using this general process:
1. Merge data
2. Split data into K-Folds
3. For each fold:
    - Fill missing values on train and test data
    - Fit model to train set
    - Make predictions on test set
    - Evaluate predictions
4. Record/Log metrics

The algorithms used were LightGBM and XGBoost for their scalability. Initial model parameters were chosen based on documentation for Best Accuracy as well as Kaggle kernels **LINK**. Ultimately, the goal was to minimize the RMSLE error on the test subset. New features were generated from the `timestamp` variable, specifically, day, month, hour, and day of the week were extracted. To avoid data leakage, filling missing values was accomplished during cross-validation so as to avoid using aggregated statistics from the entire feature space *X* within the training data. Evaluation metrics were saved to a *csv* file containg test set values of RMSLE, RMSE, and MAE. Addtionally, a plot was produced displaying feature importance as determined by the algorithm. 


### Refinement
The first model did not contain any additional features, outliers were not removed, and the target was simply the `meter_reading`. Initial metrics were not impressive, RMSLE was scored in the range of 1.5 to 2.0. To improve the score, new features were added and outliers were removed. With two steps the RMSLE dropped to a range of 0.85 to 1.0. Next steps involved transforming the target variable using the natural logarithm since the distribution appeared to be exponential. The new target became ln(`meter_reading`). Transforming the target proved to be successful as the RMSLE was further reduced to the range of 0.25 to 0.35.

Accuracy is an important aspect of the model, however, it is also important to overcome overfitting. If the model is given an infinite amount of training data, then it will keep improving constantly. However, it is not the goal to continuosly improve the training score. Minimizing the error on the test subset is the real goal. Therefore, the model needs regularization. Overfitting was overcome the adjusting the hyper-parameters of the model. [2] The hyper-parameters control the complexity and regularization of the model. As complexity of the model increases, regularization will add to the penalty (error) such that the model will try to capture the general trend and not the noise. Specific hyper-parameters will be discussed in the "Results" section. 

## Results
### Model Evaluation and Validation
KFold cross-validation was used to validate the model's results. LightGBM was used to create the final model as it performed better than XGBoost, that is, it offered a lower error on the test subset with k-fold cross-validation. 
### Justification


## Conclusion
### Reflection
### Improvement
There are numerous other methods which may improve the model further. 


### Sources
1. "ASHRAE - Great Energy Predictor III". Competition Description. https://www.kaggle.com/c/ashrae-energy-prediction
2. "Parameters Tuning - LightGBM". https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
