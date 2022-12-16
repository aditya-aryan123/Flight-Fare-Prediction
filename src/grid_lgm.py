import regex as re
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection

import lightgbm as lgm


def grid_search_lgm():
    df = pd.read_csv('../input/final_frame.csv')

    df.drop(['Date_of_Journey', 'Route', 'Dep_Date', 'Arrival_Date'], axis=1, inplace=True)

    label_features = ['Additional_Info', 'City1',
                      'City2', 'City3', 'City4', 'City5', 'City6', 'holiday', 'holiday_type']
    for col in label_features:
        le = preprocessing.LabelEncoder()
        df.loc[:, col] = le.fit_transform(df[col])

    ordinal_features = ['Dep_Type_of_Day', 'Arr_Type_of_Day', 'Dep_WeekName', 'Arr_WeekName']
    for col in ordinal_features:
        oe = preprocessing.OrdinalEncoder()
        df.loc[:, col] = oe.fit_transform(df[col].values.reshape(-1, 1))

    dummy_features = ['Airline', 'Source', 'Destination', 'Flight_Dep_Type', 'Flight_Arr_Type']
    df = pd.get_dummies(data=df, columns=dummy_features)

    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                  df.columns]

    train = df.drop('Price', axis=1)
    test = df['Price']

    X_train, X_test, y_train, y_test = model_selection.train_test_split(train, test, test_size=0.2)

    model = lgm.LGBMRegressor(n_estimators=500, early_stopping_rounds=20)

    params = {'max_depth': [10],
              'learning_rate': [0.05],
              'min_data_in_leaf': [2],
              'min_gain_to_split': [0.7],
              'min_sum_hessian_in_leaf': [2.0],
              'num_leaves': [20],
              'lambda_l1': [0, 10, 50, 100],
              'lambda_l2': [0, 10, 50, 100],
              'bagging_fraction': np.arange(0.0, 1.0, 0.1),
              'bagging_freq': np.arange(1, 10, 1)}

    grid = model_selection.GridSearchCV(model, params, cv=3, verbose=5, n_jobs=-1,
                                        scoring='neg_root_mean_squared_error')
    grid.fit(X=X_train, y=y_train, eval_set=(X_test, y_test))
    print(grid.score(X_test, y_test))
    print(grid.best_params_)


print(grid_search_lgm())
