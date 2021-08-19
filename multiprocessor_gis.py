# packages
import pandas as pd
from tqdm.notebook import tqdm
from geopy import distance
import numpy as np

# builtins
import sys
import os
import gc
from multiprocessing import Pool


def remove_category_duplicates(input):
    """Multiprocessing service for remove_duplicates"""

    df, max_dist_dup = input
    df_inner = df.copy()

    for index, row in df.iterrows():
        for index_inner, row_inner in df_inner.iterrows():
            # make sure rows do not check with themselves
            if not index == index_inner:
                coords_outer = row.loc["latitude":"longitude"]
                coords_inner = row_inner.loc["latitude":"longitude"]

                if distance.distance(coords_outer, coords_inner).m < max_dist_dup:
                    # remove row, since duplicate has been found
                    df.drop([index], axis=0, inplace=True)
                    df_inner.drop([index_inner], axis=0, inplace=True)
                    break  # row has now been removed, finding another duplicate won't change anything

    gc.collect()  # memory management
    return df


def remove_duplicates(df, cores, max_dist_dup):
    """Removes duplicates from df"""

    partitions = [df[df["asset_type"] == category].copy() for category in pd.unique(df["asset_type"])]
    partitions = [(df, max_dist_dup) for df in partitions]

    pool = Pool(processes=cores)
    results = []

    # Run partitions in parallel
    for x in tqdm(pool.imap_unordered(remove_category_duplicates, partitions), total=len(partitions)):
        results.append(x)

    pool.close()
    pool.join()

    return pd.concat(results)


def calc_dist(row):
    """Calculate distance between two points in row."""

    coords_gis = row.loc["latitude":"longitude"]
    coords_property = row.loc["prop_lat":"prop_long"]

    return distance.distance(coords_gis, coords_property).m


def create_dist_cols(row, col, df_gis_filtered, count_radius):
    """Returns row with added distance based features"""

    # create new dataframe with dist to every gis location
    df_gis_copy = df_gis_filtered.loc[:, "latitude":"longitude"].copy()
    df_gis_copy["prop_lat"] = row["latitude"]
    df_gis_copy["prop_long"] = row["longitude"]
    df_gis_copy["dist"] = df_gis_copy.apply(calc_dist, axis=1)  # calculate distance to every POI from gis df

    count = df_gis_copy[df_gis_copy["dist"] < count_radius].shape[0]  # count closer than count_radius m
    closest = df_gis_copy["dist"].min()  # minimum distance to object for this category

    row[col + "_count"] = count  # add mean of k shortest distances
    row[col + "_dist"] = closest  # add shortest distance to col

    return row


def add_dist_features(input):
    """Multiprocessing service for process_dist_features"""

    df_structured, df_gis, count_radius = input

    for col in pd.unique(df_gis["asset_type"]):

        df_structured[col + "_dist"] = 99999  # init _dist columns for this asset type
        df_structured[col + "_count"] = 0  # init _count columns for this asset type
        df_gis_filtered = df_gis[df_gis["asset_type"] == col]  # filter to correct asset type

        df_structured = df_structured.apply(create_dist_cols, axis=1, result_type="broadcast",
                                            args=[col, df_gis_filtered, count_radius])

    gc.collect()  # memory management
    return df_structured


def process_dist_features(gis_df, df_structured, count_radius, cores):
    """Returns structured df with added distance based features"""

    partitions = np.array_split(df_structured, 100)  # divide workload into 100 parts for multiprocessing
    partitions = [(df, gis_df, count_radius) for df in partitions]  # add reference to gis data

    pool = Pool(processes=cores)
    results = []

    # Run partitions in parallel
    for x in tqdm(pool.imap_unordered(add_dist_features, partitions), total=len(partitions)):
        results.append(x)

    pool.close()
    pool.join()

    return pd.concat(results, ignore_index=True)


# wrap execution in main method to guard multiprocessing code
if __name__ == "__main__":
    __spec__ = None  # remove spec to be able to run this script from iPython (Jupyter Notebook)
    df_path = sys.argv[1]
    cores = int(sys.argv[2])
    remove_duplicates(df_path, cores)
