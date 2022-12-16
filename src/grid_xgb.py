import numpy as np
import pandas as pd
import regex as re

from sklearn import preprocessing
from sklearn import model_selection
import xgboost as xgb


def grid_search_xgb():
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

    model = xgb.XGBRegressor()

    params = {'eta': [0.09],
              'max_depth': [11],
              'gamma': [50],
              'min_child_weight': [2],
              'reg_alpha': [0],
              'reg_lambda': [10]}

    grid = model_selection.GridSearchCV(model, params, cv=3, verbose=5, n_jobs=-1,
                                        scoring='neg_root_mean_squared_error')
    grid.fit(X_train, y_train)
    print(grid.score(X_test, y_test))
    print(grid.best_params_)


print(grid_search_xgb())
