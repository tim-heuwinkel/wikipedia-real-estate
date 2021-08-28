import numpy as np
import math
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from tqdm.notebook import tqdm


def _find_coord(x, df):
    """Returns id, latitude and longitude for property with given id"""

    _id, lat, long = x[0], x[1], x[2]
    row = df[df["_id"] == _id].iloc[0]
    return row["_id"], row["latitude"], row["longitude"]


def make_train_test_quarters(df, dummies=["MUNICODE"], verbose=True):
    """Returns train/test sets along with column names and df for saving errors"""

    to_drop = ["PROPERTYZIP", "MUNICODE", "SCHOOLCODE", "NEIGHCODE", "SALEDATE", "SALEPRICE",
               "FAIRMARKETTOTAL", "latitude", "longitude", "SALEYEAR", "DISTRICT"]
    to_drop = [x for x in to_drop if x not in dummies]

    X = df.drop(to_drop, axis=1)

    if len(dummies) > 0:
        X = pd.get_dummies(X, columns=dummies)

    # save col names for later
    X_columns = list(X.columns)
    # remove id from col list, since it will be filtered out later
    X_columns.remove("_id")

    y = df["SALEPRICE"]

    X_half_1, X_half_2, y_half_1, y_half_2 = train_test_split(
        X, y, test_size=0.5, random_state=42)

    X_quarter_1, X_quarter_2, y_quarter_1, y_quarter_2 = train_test_split(
        X_half_1, y_half_1, test_size=0.5, random_state=42)

    X_quarter_3, X_quarter_4, y_quarter_3, y_quarter_4 = train_test_split(
        X_half_2, y_half_2, test_size=0.5, random_state=42)

    # all known data
    X_train = pd.concat([X_half_1, X_quarter_3])
    y_train = list(y_half_1) + list(y_quarter_3)

    # all unknown data
    X_test = X_quarter_4.copy()
    y_test = list(y_quarter_4)

    # save ids for later
    quarter_1_ids = X_quarter_1["_id"]
    quarter_2_ids = X_quarter_2["_id"]
    quarter_3_ids = X_quarter_3["_id"]
    quarter_4_ids = X_quarter_4["_id"]

    # remove first column (id)
    X_quarter_1 = X_quarter_1.drop(["_id"], axis=1)
    X_quarter_2 = X_quarter_2.drop(["_id"], axis=1)
    X_quarter_3 = X_quarter_3.drop(["_id"], axis=1)
    X_quarter_4 = X_quarter_4.drop(["_id"], axis=1)

    # remove unknown levels in test
    cols = X_train.columns
    for col in cols:
        if "_" in col and len(pd.unique(X_train[col])) <= 2:
            if np.sum(X_train[col]) == 0:  # no observations from this level
                if verbose:
                    print(f"removed column {col}, first occurence in test")
                X_quarter_1.drop([col], axis=1, inplace=True)
                X_quarter_2.drop([col], axis=1, inplace=True)
                X_quarter_3.drop([col], axis=1, inplace=True)
                X_quarter_4.drop([col], axis=1, inplace=True)

                X_train.drop([col], axis=1, inplace=True)
                X_test.drop([col], axis=1, inplace=True)
                X_columns.remove(col)

    return X_columns, [X, X_quarter_1, X_quarter_2, X_quarter_3, X_quarter_4], [y, y_quarter_1, y_quarter_2,
                                                                                y_quarter_3, y_quarter_4], [
               quarter_1_ids, quarter_2_ids, quarter_3_ids, quarter_4_ids]


def make_train_test(df, dummies=["MUNICODE"], verbose=True, test_size=0.25):
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
        X, y, test_size=test_size, random_state=42)

    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=42)

    # remove unknown levels in test
    cols = X_train.columns
    for col in cols:
        if "_" in col and len(pd.unique(X_train[col])) <= 2:
            if np.sum(X_train[col]) == 0:  # no observations from this level
                if verbose:
                    print(f"removed column {col}, first occurence in test")
                X_train.drop([col], axis=1, inplace=True)
                X_train_train.drop([col], axis=1, inplace=True)
                X_train_val.drop([col], axis=1, inplace=True)
                X_test.drop([col], axis=1, inplace=True)
                X_columns.remove(col)

    # save ids for later
    train_ids = X_train["_id"]
    test_ids = X_test["_id"]
    X_train = X_train.drop(["_id"], axis=1)  # remove first column (id)
    X_test = X_test.drop(["_id"], axis=1)  # remove first column (id)

    if verbose:
        print("")
        print(f"{X_train.shape}: {X_train_train.shape} + {X_train_val.shape}")
        print(f"{y_train.shape}: {y_train_train.shape} + {y_train_val.shape}")
        print(X_test.shape)
        print(y_test.shape)

    # create error df
    error_df = pd.DataFrame(
        data={"id": test_ids, "lat": [0] * len(test_ids), "long": [0] * len(test_ids)})
    error_df = error_df.apply(lambda x: _find_coord(
        x, df), axis=1, result_type='broadcast')

    return X_columns, [X, y, X_train, X_test, y_train, y_test, X_train_train, X_train_val, y_train_train,
                       y_train_val], error_df


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


def get_group_importance(fi, cols):

    struct_end = -1
    text_end = -1
    for i, word in enumerate(cols):
        if "ROOFDESC_" in word:
            struct_end = i + 1
        elif "MUNICODE" in word:
            text_end = i
            break
    
    struct_fi = sum(fi[:struct_end])

    if text_end == -1:
        text_fi = sum(fi[struct_end:])
        return [struct_fi, text_fi]
    else:
        text_fi = sum(fi[struct_end:text_end])
        muni_fi = sum(fi[text_end:])
        return [struct_fi, text_fi, muni_fi]


def cross_validation(estimator, X, y, k_folds, additional_drops=[], verbose_drop=True, return_std=False):
    """Returns and prints cross validated MAE, RMSE, MAPE and R^2"""

    maes, rmses, mapes, r_squareds = [], [], [], []
    fis = []
    X_cv = X.drop(["_id"], axis=1)  # remove "_id" column
    X_cv_cols = X.columns

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for train_index, test_index in tqdm(kf.split(X_cv), total=5):
        X_train, X_test = X_cv.iloc[train_index], X_cv.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train.drop(additional_drops, axis=1, inplace=True)
        X_test.drop(additional_drops, axis=1, inplace=True)

        # remove unknown levels in test
        X_cv_cols = list(X_train.columns)
        for col in X_cv_cols:
            if "_" in col and len(pd.unique(X_train[col])) <= 2:  # is binary column
                if np.sum(X_train[col]) == 0:  # no observations from this level
                    if verbose_drop:
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
            fis.append([])
        else:
            estimator.fit(X=X_train, y=y_train, verbose=False)
            fis_current = get_group_importance(estimator.get_feature_importance(), X_cv_cols)
            fis.append(fis_current)

        y_pred_cv = estimator.predict(X_test)
        mae, rmse, mape, r_squared = get_metrics(y_test, y_pred_cv, print_out=False)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        r_squareds.append(r_squared)

    mae_cv, rmse_cv = round(np.mean(maes)), round(np.mean(rmses))
    mape_cv, r_squared_cv = round(np.mean(mapes), 2), round(np.mean(r_squareds), 3)
    fis_cv = np.mean(fis, axis=0)

    print("")
    print(f"MAE:  {mae_cv}")
    print(f"RMSE: {rmse_cv}")
    print(f"MAPE: {mape_cv}%")
    print(f"R^2:  {r_squared_cv}")

    return [mae_cv, rmse_cv, mape_cv, r_squared_cv], X_cv_cols


def soos_validation(estimator, df, additional_drops=[], verbose_drop=True, standardize=False):
    soos_df = df.copy()
    soos_df = soos_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle data
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
    fis = []

    for i in range(1, 14):
        print(f"Predicting district {i}/13")

        train = soos_df[soos_df["DISTRICT"] != "district_" + str(i)]  # leave out i'th district
        test = soos_df[soos_df["DISTRICT"] == "district_" + str(i)]

        # train = train.drop(["_id", "PROPERTYZIP", "SCHOOLCODE", "NEIGHCODE", "SALEDATE",
        #                     "FAIRMARKETTOTAL", "latitude", "longitude", "SALEYEAR", "DISTRICT"], axis=1)
        # test = test.drop(["_id", "PROPERTYZIP", "SCHOOLCODE", "NEIGHCODE", "SALEDATE",
        #                   "FAIRMARKETTOTAL", "latitude", "longitude", "SALEYEAR", "DISTRICT"], axis=1)
        to_drop = ["_id", "PROPERTYZIP", "SCHOOLCODE", "NEIGHCODE", "SALEDATE", "MUNICODE",
                   "FAIRMARKETTOTAL", "latitude", "longitude", "SALEYEAR", "DISTRICT"]
        train = train.drop(to_drop, axis=1)
        test = test.drop(to_drop, axis=1)

        # drop additional columns if given
        train = train.drop(additional_drops, axis=1)
        test = test.drop(additional_drops, axis=1)

        # remove unknown levels in test
        X_cv_cols = list(train.columns)
        for col in X_cv_cols:
            if "_" in col and len(pd.unique(train[col])) <= 2:  # is binary column
                if np.sum(train[col]) == 0:  # no observations from this level
                    if verbose_drop:
                        print(f"removed column {col}, first occurence in test")
                    train.drop([col], axis=1, inplace=True)
                    test.drop([col], axis=1, inplace=True)
                    X_cv_cols.remove(col)

        X_train = train.drop(["SALEPRICE"], axis=1)
        col_names = X_train.columns
        X_train = X_train.to_numpy()
        y_train = train["SALEPRICE"].to_numpy()

        X_test = test.drop(["SALEPRICE"], axis=1).to_numpy()
        y_test = test["SALEPRICE"].to_numpy()

        if standardize:
            train_mean = X_train.mean()
            train_std = X_train.std()

            X_train = (X_train - train_mean) / train_std
            X_test = (X_test - train_mean) / train_std

        if "linear_model" in str(type(estimator)):
            estimator.fit(X=X_train, y=y_train)
            fis.append(np.array(estimator.coef_))
        else:
            estimator.fit(X=X_train, y=y_train, verbose=False)
            fis.append(estimator.get_feature_importance())

        y_pred_cv = estimator.predict(X_test)
        y_preds.extend(y_pred_cv)
        errors.extend([test - pred for test, pred in zip(y_test, y_pred_cv)])

        mae, rmse, mape, r_squared = get_metrics(y_test, y_pred_cv, print_out=False)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        r_squareds.append(r_squared)

    error_df_soos["prediction"] = y_preds
    error_df_soos["error"] = errors

    all_sum = error_df_soos.shape[0]
    weights = [error_df_soos[error_df_soos["district"] == "district_" + str(i)].shape[0] / all_sum for i in
               range(1, 14)]

    avg_mae = sum(np.multiply(maes, weights))
    avg_rmse = sum(np.multiply(rmses, weights))
    avg_mape = sum(np.multiply(mapes, weights))
    avg_r2 = sum(np.multiply(r_squareds, weights))

    weighted_fis = [fi * weight for fi, weight in zip(fis, weights)]
    avg_fi = np.array(weighted_fis).sum(axis=0)

    print("")
    print("Weighted metrics:")
    print(f"MAE:  {round(avg_mae)}")
    print(f"RMSE: {round(avg_rmse)}")
    print(f"MAPE: {round(avg_mape, 2)}%")
    print(f"R^2:  {round(avg_r2, 3)}")

    return error_df_soos, col_names, avg_fi, [maes, rmses, mapes, r_squareds]
