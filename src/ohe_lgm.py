import pandas as pd
import regex as re
import shap
import matplotlib.pyplot as plt

from yellowbrick.regressor import prediction_error

from sklearn import model_selection
from sklearn import metrics

import lightgbm as lgbm


def run():
    df = pd.read_csv('../input/final_frame.csv')

    df.drop(['Dep_Month', 'Dep_Day', 'Dep_Week', 'Dep_Quarter', 'Dep_Time_Hour', 'Dep_Time_Minute', 'Arr_Month',
             'Arr_Day', 'Date_of_Journey', 'Route', 'Dep_Date', 'Arrival_Date', 'Arr_Week', 'Arr_Quarter',
             'Arrival_Time_Hour', 'Arrival_Time_Minute'], axis=1, inplace=True)

    categorical_columns = [col for col in df.columns if df[col].dtypes == 'object']
    df = pd.get_dummies(data=df, columns=categorical_columns)

    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                  df.columns]

    X = df.drop('Price', axis=1)
    y = df['Price']

    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = lgbm.LGBMRegressor(num_iterations=1000, max_depth=10, min_gain_to_split=0.1, min_data_in_leaf=1,
                               min_sum_hessian_in_leaf=0.0, learning_rate=0.1)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    r2_score_train = metrics.r2_score(y_train, pred_train)
    rmse_train = metrics.mean_squared_error(y_train, pred_train, squared=False)
    r2_score_test = metrics.r2_score(y_test, pred_test)
    rmse_test = metrics.mean_squared_error(y_test, pred_test, squared=False)
    print(f"Root Mean Squared Error Train: {rmse_train}, R Squared Train: {r2_score_train}")
    print(f"Root Mean Squared Error Test: {rmse_test}, R Squared Test: {r2_score_test}")

    visualizer = prediction_error(model, X_train, y_train, X_test, y_test)
    print(visualizer)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, color=plt.get_cmap("tab20c"))


print(run())
