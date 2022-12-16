import pandas as pd
from sklearn import metrics
import argparse
import model_dispatcher


def run(fold, model):
    df = pd.read_csv('../input/final_frame_folds.csv')
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    x_train = df_train.drop('Price', axis=1).values
    y_train = df_train.Price.values
    x_valid = df_valid.drop('Price', axis=1).values
    y_valid = df_valid.Price.values
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)
    r2 = metrics.r2_score(y_valid, preds)
    mae = metrics.mean_absolute_error(y_valid, preds)
    rmse = metrics.mean_squared_error(y_valid, preds, squared=False)
    print(f"Fold={fold}, Root Mean Squared Error={rmse}, R Squared={r2}, Mean Absolute Error={mae}, Model={clf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()
    run(
        fold=args.fold,
        model=args.model
    )
