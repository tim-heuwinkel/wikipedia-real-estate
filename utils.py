import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from tqdm.notebook import tqdm


def _find_coord(x, df):
    """Returns id, latitude and longitude for property with given id"""

    _id, lat, long = x[0], x[1], x[2]
    row = df[df["_id"] == _id].iloc[0]
    return row["_id"], row["latitude"], row["longitude"]


def make_train_test(df):
    """Returns train/test sets along with column names and df for saving errors"""

    X = df.drop(["PROPERTYZIP", "MUNICODE", "SCHOOLCODE", "NEIGHCODE", "SALEDATE", "SALEPRICE",
                 "FAIRMARKETTOTAL", "latitude", "longitude", "SALEYEAR"], axis=1)

    # save col names for later
    X_columns = list(X.columns)
    # remove id from col list, since it will be filtered out later
    X_columns.remove("_id")
    X = X.to_numpy()

    y = df["SALEPRICE"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # save ids for later
    train_ids = [x[0] for x in X_train]
    test_ids = [x[0] for x in X_test]
    X_train = X_train[:, 1:]  # remove first column (id)
    X_test = X_test[:, 1:]    # remove first column (id)

    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42)

    print(f"{X_train.shape}: {X_train_train.shape} + {X_train_val.shape}")
    print(f"{y_train.shape}: {y_train_train.shape} + {y_train_val.shape}")
    print(X_test.shape)
    print(y_test.shape)

    # create error df
    error_df = pd.DataFrame(
        data={"id": test_ids, "lat": [0]*len(test_ids), "long": [0]*len(test_ids)})
    error_df = error_df.apply(lambda x: _find_coord(
        x, df), axis=1, result_type='broadcast')
    error_df.head(10)

    return X_columns, [X, y, X_train, X_test, y_train, y_test, X_train_train, X_train_val, y_train_train, y_train_val], error_df


def mean_absolute_percentage_error(y_true, y_pred):
    """Returns MAPE"""

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_metrics(y_true, y_pred, print_out=True):
    """Returns MAE, RMSE, MAPE and R^2"""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)

    if print_out:
        print(f"MAE:  {round(mae)}")
        print(f"RMSE: {round(rmse)}")
        print(f"MAPE: {round(mape, 2)}%")
        print(f"R^2:  {round(r_squared, 3)}")

    return mae, rmse, mape, r_squared


def cross_validation(estimator, X, y, k_folds):
    """Returns and prints cross validated MAE, RMSE, MAPE and R^2"""

    maes, rmses, mapes, r_squareds = [], [], [], []
    X_cv = X[:, 1:]  # remove "_id" column

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for train_index, test_index in tqdm(kf.split(X_cv), total=5):
        X_train, X_test = X_cv[train_index], X_cv[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if "linear_model" in str(type(estimator)):
            estimator.fit(X=X_train, y=y_train)
        else:
            estimator.fit(X=X_train, y=y_train, verbose=False)

        y_pred_cv = estimator.predict(X_test)
        mae, rmse, mape, r_squared = get_metrics(y_test, y_pred_cv, print_out=False)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        r_squareds.append(r_squared)

    mae_cv, rmse_cv = round(np.mean(maes)), round(np.mean(rmses))
    mape_cv, r_squared_cv = round(np.mean(mapes), 2), round(np.mean(r_squareds), 3)

    print(f"MAE:  {mae_cv}")
    print(f"RMSE: {rmse_cv}")
    print(f"MAPE: {mape_cv}%")
    print(f"R^2:  {r_squared_cv}")

    return mae_cv, rmse_cv, mape_cv, r_squared_cv

