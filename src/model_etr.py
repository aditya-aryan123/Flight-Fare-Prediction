import pandas as pd

import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import ensemble

from yellowbrick.regressor import prediction_error


def run():
    df = pd.read_csv('../input/final.csv')

    X = df.drop('Price', axis=1)
    y = df['Price']

    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = ensemble.ExtraTreesRegressor(n_estimators=1000, max_depth=20, max_leaf_nodes=50, min_impurity_decrease=7.9)
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


print(run())
