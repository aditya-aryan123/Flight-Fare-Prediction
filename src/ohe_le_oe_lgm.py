import pandas as pd
import regex as re
import shap
import matplotlib.pyplot as plt

from yellowbrick.regressor import prediction_error

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import lightgbm as lgbm


def run():
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

    X = df.drop('Price', axis=1)
    y = df['Price']

    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = lgbm.LGBMRegressor(max_depth=20)
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

    booster = model.booster_
    df_feature_importance = (
        pd.DataFrame({
            'feature': booster.feature_name(),
            'importance': booster.feature_importance(importance_type='gain'),
        })
        .sort_values('importance', ascending=False)
    )
    df_feature_importance.to_csv('LightGBM_Feature_Importance.csv', index=False)


print(run())
