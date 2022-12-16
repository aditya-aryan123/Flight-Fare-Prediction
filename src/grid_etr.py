import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn import ensemble


def grid_search_xgb():
    df = pd.read_csv('../input/final.csv')

    train = df.drop('Price', axis=1)
    test = df['Price']

    X_train, X_test, y_train, y_test = model_selection.train_test_split(train, test, test_size=0.2)

    model = ensemble.ExtraTreesRegressor()

    params = {'max_depth': [None, 5, 10, 15, 20],
              'max_leaf_nodes': [15, 20, 25, 30, 35, 40, 45, 50],
              'min_samples_split': [1, 2, 3, 4, 5],
              'min_samples_leaf': [1],
              'min_impurity_decrease': [7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0]}

    grid = model_selection.GridSearchCV(model, params, cv=3, verbose=5, n_jobs=-1,
                                        scoring='neg_root_mean_squared_error')
    grid.fit(X_train, y_train)
    print(grid.score(X_test, y_test))
    print(grid.best_params_)


print(grid_search_xgb())
