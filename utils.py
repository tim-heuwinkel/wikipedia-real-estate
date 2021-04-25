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


def make_train_test(df, dummies=["MUNICODE"]):
    """Returns train/test sets along with column names and df for saving errors"""

    to_drop = ["PROPERTYZIP", "MUNICODE", "SCHOOLCODE", "NEIGHCODE", "SALEDATE", "SALEPRICE",
               "FAIRMARKETTOTAL", "latitude", "longitude", "SALEYEAR", "DISTRICT"]
    to_drop = [x for x in to_drop if not x in dummies]

    X = df.drop(to_drop, axis=1)

    if len(dummies) > 0:
        X = pd.get_dummies(X, columns=dummies)

    # save col names for later
    X_columns = list(X.columns)
    # remove id from col list, since it will be filtered out later
    X_columns.remove("_id")

    y = df["SALEPRICE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # remove unknown levels in test
    cols = X_train.columns
    for col in cols:
        if "_" in col and len(pd.unique(X_train[col])) <= 2:
            if np.sum(X_train[col]) == 0:  # no observations from this level
                print(f"removed column {col}, first occurence in test")
                X_train.drop([col], axis=1, inplace=True)
                X_test.drop([col], axis=1, inplace=True)
                X_columns.remove(col)

    # save ids for later
    train_ids = X_train["_id"]
    test_ids = X_test["_id"]
    X_train = X_train.drop(["_id"], axis=1)  # remove first column (id)
    X_test = X_test.drop(["_id"], axis=1)    # remove first column (id)

    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42)

    print("")
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
    X_cv = X.drop(["_id"], axis=1)  # remove "_id" column
    # X_cv = X_cv.to_numpy()
    X_cv_cols = X.columns

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for train_index, test_index in tqdm(kf.split(X_cv), total=5):
        X_train, X_test = X_cv.iloc[train_index], X_cv.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # remove unknown levels in test
        X_cv_cols = list(X_train.columns)
        for col in X_cv_cols:
            if "_" in col and len(pd.unique(X_train[col])) <= 2:  # is binary column
                if np.sum(X_train[col]) == 0:  # no observations from this level
                    print(f"removed column {col}, first occurence in test")
                    X_train.drop([col], axis=1, inplace=True)
                    X_test.drop([col], axis=1, inplace=True)
                    X_cv_cols.remove(col)

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

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

    print("")
    print(f"MAE:  {mae_cv}")
    print(f"RMSE: {rmse_cv}")
    print(f"MAPE: {mape_cv}%")
    print(f"R^2:  {r_squared_cv}")

    return [mae_cv, rmse_cv, mape_cv, r_squared_cv], X_cv_cols


def soos_validation(estimator, df):
    soos_df = df.copy()
    soos_df = soos_df.sample(frac=1).reset_index(drop=True)  # shuffle data
    soos_df.sort_values(by=["DISTRICT"])  # sort by district

    error_df_soos = pd.DataFrame(
        data={"id": soos_df["_id"],
              "lat": soos_df["latitude"],
              "long": soos_df["longitude"],
              "district": soos_df["DISTRICT"],
              "prediction": 0,
              "error": 0})

    y_preds = []
    errors = []
    maes, rmses, mapes, r_squareds = [], [], [], []

    for i in range(1, 14):
        train = soos_df[soos_df["DISTRICT"] != "district_"+str(i)]  # leave out i'th district
        test = soos_df[soos_df["DISTRICT"] == "district_"+str(i)]

        train = train.drop(["_id", "PROPERTYZIP", "SCHOOLCODE", "NEIGHCODE", "SALEDATE",
                            "FAIRMARKETTOTAL", "latitude", "longitude", "SALEYEAR", "DISTRICT"], axis=1)
        test = test.drop(["_id", "PROPERTYZIP", "SCHOOLCODE", "NEIGHCODE", "SALEDATE",
                          "FAIRMARKETTOTAL", "latitude", "longitude", "SALEYEAR", "DISTRICT"], axis=1)

        X_train = train.drop(["SALEPRICE"], axis=1).to_numpy()
        y_train = train["SALEPRICE"].to_numpy()

        X_test = test.drop(["SALEPRICE"], axis=1).to_numpy()
        y_test = test["SALEPRICE"].to_numpy()

        if "linear_model" in str(type(estimator)):
            estimator.fit(X=X_train, y=y_train)
        else:
            estimator.fit(X=X_train, y=y_train, verbose=False)

        y_pred_cv = estimator.predict(X_test)
        y_preds.extend(y_pred_cv)
        errors.extend([test - pred for test, pred in zip(y_test, y_pred_cv)])

        mae, rmse, mape, r_squared = get_metrics(y_test, y_pred_cv, print_out=False)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        r_squareds.append(r_squared)

        print(f"Predicting district {i}/13")

    error_df_soos["prediction"] = y_preds
    error_df_soos["error"] = errors

    all_sum = error_df_soos.shape[0]
    weights = [error_df_soos[error_df_soos["district"]=="district_"+str(i)].shape[0] / all_sum for i in range(1, 14)]

    avg_mae = sum(np.multiply(maes, weights))
    avg_rmse = sum(np.multiply(rmses, weights))
    avg_mape = sum(np.multiply(mapes, weights))
    avg_r2 = sum(np.multiply(r_squareds, weights))

    # avg_mae = np.mean(maes)
    # avg_rmse = np.mean(rmses)
    # avg_mape = np.mean(mapes)
    # avg_r2 = np.mean(r_squareds)
    print("")
    print("Weighted metrics:")
    print(f"MAE:  {round(avg_mae)}")
    print(f"RMSE: {round(avg_rmse)}")
    print(f"MAPE: {round(avg_mape, 2)}%")
    print(f"R^2:  {round(avg_r2, 3)}")

    return error_df_soos, [maes, rmses, mapes, r_squareds]
