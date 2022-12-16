from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

models = {
    'decision_tree': tree.DecisionTreeRegressor(),
    'random_forest': ensemble.RandomForestRegressor(),
    'linear_reg': linear_model.LinearRegression(),
    'lasso': linear_model.Lasso(),
    'ridge': linear_model.Ridge(),
    'etr': ensemble.ExtraTreesRegressor(),
    'gbr': ensemble.GradientBoostingRegressor(),
    'hgbr': ensemble.HistGradientBoostingRegressor(),
    'abr': ensemble.AdaBoostRegressor(),
    'gnb': naive_bayes.GaussianNB(),
    'svm': svm.SVR(),
    'xgb': xgb.XGBRegressor(),
    'lgm': lgb.LGBMRegressor(),
    'cat': cat.CatBoostRegressor()
}
