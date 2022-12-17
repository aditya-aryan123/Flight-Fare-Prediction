# Flight-Fare-Prediction

Flight ticket prices can be something hard to guess, today we might see a price, check out the price of the same flight tomorrow, it will be a different story. We might have often heard travelers saying that flight ticket prices are so unpredictable. As data scientists, we are gonna prove that given the right data anything can be predicted. Here you will be provided with prices of flight tickets for various airlines for the year of 2019 and between various cities. Size of training set: 10683 records. Our goal here is to create a machine learning model for predicting flight ticket price.


## Business Problem

The flight ticket price in India is based on demand and supply model with few restrictions on pricing from regulatory bodies. It is often perceived as unpredictable and , recent dynamic pricing scheme added to the confusion. The objective is to create a machine learning model for predicting the flight price, based on historical data, which can be used for reference price for customers as well as airline service providers


## My view on the Problem

The training set contains the features, along with the prices of the flights. It contains 10683 records, 10 input features and 1 output column — ‘Price’. Basic EDA revealed there were ~0.009361% missing values in Route and Total_Stops column, so I dropped those rows. Extracted new date time features from the date column. Performed string operations to extract meaningful features. Flight Prices are affected by holidays and the distance. I downloaded India holiday data from Kaggle and for Latitude and Longitude I used Indian Cities Database from Kaggle and calculated the distance between two cities using Haversine formula. After that I applies cosine and sine transformation for cyclical features like hour, month and day, week, quarter, minute. Then I calculated Variance Inflation factor using statsmodels library to find multi-collinearity between variables. There were multiple columns which shows multi-collinearity. I used statsmodels OLS to check the relation between variable and their impact on target variable. I didn't drop any multi-collinear columns as the linear model performed better with those features. I experimented with the basic algorithms and found Extra Tree Regressor, and Random Forest Regressor to give the lowest root mean squared error. After that I tried XGBoost Regressor and Light Gradient Boosting Regressor. My experimentations revealed that the LightGBM model seemed to perform well. Therefore, I went ahead with the said model to test which of the hyperparameters gave the highest score as per the ground truth. Finally, I scripted the entire process in a modular fashion to create a pipeline that could be deployed and automated for future use.


## Folder Structure

Input Folder contains dataset used.

Notebook Folder contains Jupyter Notebooks.

src Folder contains models and grid_search files.


## File Structure

create_folds.py: Used to create k-folds for cross validation purpose.

train.py: Used to train multiple models 

gridsearch_name.py: Used for grid search. Select best hyperparameter.

model_name.py: Model used


## Final Result

LightGBM gave the best result.

Train Set: Root mean squared error: 600.76, Mean absolute error: 399.28, R Squared: 0.982

Test Set: Root mean squared error: 1428.74, Mean absolute error: 723.85, R Squared: 0.907
