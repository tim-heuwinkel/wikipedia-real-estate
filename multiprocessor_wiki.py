# packages
import xml.sax  # parse xml
import numpy as np
import pandas as pd
import mwparserfromhell  # parse wikimedia
from tqdm.notebook import tqdm  # progress bars
from geopy import distance
from data_processor import *  # parse coordinates

# builtins
import os
import sys
import bz2
import gc
import json
from multiprocessing import Pool


def process_article(title, text, timestamp, coord_range_lat, coord_range_long, template="coord"):
    """Process wikipedia article looking for templates and returning them"""

    # Create a parsing object
    wikicode = mwparserfromhell.parse(text)

    # Search through templates for the template
    coord_matches = wikicode.filter_templates(matches=template)

    # Filter out errant matches
    coord_matches = [x for x in coord_matches if x.name.strip_code().strip().lower() == template.lower()]

    # check if match contains coordinates
    if len(coord_matches) >= 1:

        # extract coordinates
        coords = extract_coordinates(str(coord_matches[0]))

        # coords have wrong format
        if not coords:
            return None

        # check if coordinates are in Western Pennsylvania region
        if coord_range_lat[0] < coords[0] < coord_range_lat[1] and coord_range_long[0] < coords[1] < coord_range_long[1]:

            # Extract all templates
            all_templates = wikicode.filter_templates()

            infobox = [x for x in all_templates if "infobox" in x.name.strip_code().strip().lower()]

            if len(infobox) >= 1:
                # Extract information from infobox if existing
                properties = {param.name.strip_code().strip(): param.value.strip_code().strip()
                              for param in infobox[0].params
                              if param.value.strip_code().strip()}
            else:
                properties = None

            text = wikicode.strip_code().strip()

            # Extract internal wikilinks
            wikilinks = [x.title.strip_code().strip() for x in wikicode.filter_wikilinks()]

            # Extract external links
            exlinks = [x.url.strip_code().strip() for x in wikicode.filter_external_links()]

            # Find approximate length of article
            text_length = len(wikicode.strip_code().strip())

            return [title, coords, properties, text, wikilinks, exlinks, text_length]
        else:
            # object not in Western Pennsylvania region, disregard
            return None


class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Parse through XML data using SAX"""

    def __init__(self, usa):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._articles = []
        self._article_count = 0
        self._non_matches = []
        self._usa = usa

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text', 'timestamp'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            self._article_count += 1

            if self._usa:
                # rough coordinates of the USA
                COORD_RANGE_LAT = (25.11667, 49.040000)
                COORD_RANGE_LONG = (-125.666666, -59.815000)
            else:
                # rough coordinates of Allegheny region
                COORD_RANGE_LAT = (40.000000, 40.870000)
                COORD_RANGE_LONG = (-80.550000, -79.500000)

            # Search through the page to see if the coordinate is in the Allegheny region
            article = process_article(**self._values, coord_range_lat=COORD_RANGE_LAT,
                                      coord_range_long=COORD_RANGE_LONG)
            # Append to the list of articles
            if article:
                self._articles.append(article)


def find_articles(input, limit=None, save=True):
    """Find all the articles in specified region from a compressed wikipedia XML dump.
       `limit` is an optional argument to only return a set number of articles.
        If save, articles are saved to a partition directory based on file name"""

    data_path, usa = input

    # Object for handling xml
    handler = WikiXmlHandler(usa)

    # Parsing object
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    # Iterate through compressed file
    for i, line in enumerate(bz2.BZ2File(data_path, 'r')):
        try:
            parser.feed(line)
        except StopIteration:
            break

        # Optional limit
        if limit is not None and len(handler._articles) >= limit:
            return handler._articles

    if save:
        if usa:
            partition_dir = os.path.dirname(os.path.dirname(data_path)) + "\\uncompressed_usa\\"
        else:
            partition_dir = os.path.dirname(os.path.dirname(data_path)) + "\\uncompressed\\"

        if not os.path.exists(partition_dir):
            os.mkdir(partition_dir)

        # Create file name based on partition name
        p_str = data_path.split('-')[-1].split('.')[-2]
        out_dir = partition_dir + f'{p_str}.ndjson'

        # Open the file
        with open(out_dir, 'w') as fout:
            # Write as json
            for article in handler._articles:
                fout.write(json.dumps(article) + '\n')
        print(f'{len(os.listdir(partition_dir))} files processed.', end='\r')

    # Memory management
    del handler
    del parser
    gc.collect()
    return None


def process(compressed_path, usa, cores):
    """Decompress all articles in compressed_path, look for templates and save articles fitting criteria"""

    partitions = [(compressed_path + file, usa) for file in os.listdir(compressed_path) if 'xml-p' in file]
    len(partitions), partitions[-1]

    if usa:
        uncompressed_path = os.path.dirname(os.path.dirname(compressed_path)) + "\\uncompressed_usa"
    else:
        uncompressed_path = os.path.dirname(os.path.dirname(compressed_path)) + "\\uncompressed"

    if not len(os.listdir(uncompressed_path)) > 50:  # check if already preprocessed
        pool = Pool(processes=cores)
        results = []

        # Run partitions in parallel
        for x in tqdm(pool.imap_unordered(find_articles, partitions), total=len(partitions)):
            results.append(x)

        pool.close()
        pool.join()
    else:
        print("Articles already processed.")


def create_dist_cols(row, category, df_category):
    """Returns row with added distance based features"""

    # create new dataframe with dist to every gis location
    df_category_copy = df_category.loc[:, ["coords"]].copy()
    df_category_copy["prop_coords"] = [[row["latitude"], row["longitude"]]]*df_category_copy.shape[0]
    # calculate distance from current property to all objects
    df_category_copy["dist"] = df_category_copy.apply(lambda x: distance.distance(x["coords"],
                                                                                  x["prop_coords"]).m, axis=1)

    count = df_category_copy[df_category_copy["dist"] < 2500].shape[0]  # count closer than 2.5 km
    closest = df_category_copy["dist"].min()  # minimum distance to object for this category

    row[category + "_count"] = count  # add mean of k shortest distances
    row[category + "_dist"] = closest  # add shortest distance to col

    return row


def add_dist_features(input):
    """Multiprocessing service for process_dist_features"""

    df_structured, categories_dfs = input

    for category in categories_dfs:
        df_structured[category + "_dist"] = 99999
        df_structured[category + "_count"] = 0
        df_category = categories_dfs[category]

        df_structured = df_structured.apply(create_dist_cols, axis=1, result_type="broadcast",
                                            args=[category, df_category])

    gc.collect()  # memory management
    return df_structured


def process_dist_features(df_structured, categories_dfs, cores):
    """Returns structured df with added distance based features"""

    partitions = np.array_split(df_structured, 100)  # divide dataframe into 100 parts
    partitions = [(df, categories_dfs) for df in partitions]  # add reference to category data

    pool = Pool(processes=cores)
    results = []

    # Run partitions in parallel
    for x in tqdm(pool.imap_unordered(add_dist_features, partitions), total=len(partitions)):
        results.append(x)

    pool.close()
    pool.join()

    return pd.concat(results, ignore_index=True)


def create_text_cols(row_outer, places_df, tf_matrix, feature_names, max_dist, weighted):
    """Returns row along with mean of weighted term frequencies for articles closer than max_dist for property in row"""

    in_range = []
    for index, row in places_df.iterrows():
        dist = distance.distance(row["coords"], [row_outer["latitude"], row_outer["longitude"]]).m
        if dist < max_dist:
            # article is closer than max_dist m
            if weighted:
                weight = 1 - (dist / max_dist)  # calculate weight between 0 and 1
                in_range.append(tf_matrix[index] * weight)  # multiply tfs by weight
            else:
                in_range.append(tf_matrix[index])  # <- NO WEIGHTING

    if len(in_range) < 3:
        # less than 3 articles found inside max_dist radius
        row_outer["article_count"] = len(in_range)
        in_range = [pd.NA] * len(tf_matrix[0])
        row_outer.loc[feature_names[0]:feature_names[-1]] = in_range
    else:
        # calculate mean of weighted term frequencies and assign to word columns
        article_count = len(in_range)
        row_outer.loc[feature_names[0]:feature_names[-1]] = [sum(x) / article_count for x in zip(*in_range)]
        # row_outer.loc[feature_names[0]:feature_names[-1]] = [sum(x) for x in zip(*in_range)]  # just sum
        row_outer["article_count"] = article_count

    return row_outer


def add_text_features(input):
    """Multiprocessing service for process_text_features"""

    df_structured, places_df, tf_matrix, feature_names, max_dist, weighting = input

    row_count = df_structured.shape[0]
    for word in feature_names:
        # add column for each word
        df_structured[word] = [0]*row_count

    # add column for amount of articles found
    df_structured["article_count"] = [0]*row_count

    df_structured = df_structured.apply(create_text_cols, axis=1, result_type="broadcast",
                                        args=[places_df, tf_matrix, feature_names, max_dist, weighting])

    gc.collect()  # memory management
    return df_structured


def process_text_features(df_structured, places_df, tf_matrix, feature_names, max_dist, cores, weighting):
    """Returns structured df with added text based features"""

    partitions = np.array_split(df_structured, 100)  # divide dataframe into 100 parts
    # add reference to places and tf data
    partitions = [(df, places_df, tf_matrix, feature_names, max_dist, weighting) for df in partitions]

    pool = Pool(processes=cores)
    results = []

    # Run partitions in parallel
    for x in tqdm(pool.imap_unordered(add_text_features, partitions), total=len(partitions)):
        results.append(x)

    pool.close()
    pool.join()

    return pd.concat(results, ignore_index=True)


def create_doc2vec_cols(row_outer, places_df, vector_size, max_dist):
    """Returns row along with mean of weighted doc2vec vector for articles closer than max_dist for property in row"""

    in_range = []
    for index, row in places_df.iterrows():
        dist = distance.distance(row["coords"], [row_outer["latitude"], row_outer["longitude"]]).m
        if dist < max_dist:
            # article is closer than max_dist m
            weight = 1 - (dist / max_dist)  # calculate weight between 0 and 1
            in_range.append(places_df.loc[index, "vec_1":] * weight)  # multiply tfs by weight

    if len(in_range) < 3:
        # less than 3 articles found inside max_dist radius
        row_outer["article_count"] = len(in_range)
        in_range = [pd.NA] * len(vector_size)
        row_outer.loc["vec_1":] = in_range
    else:
        # calculate mean of weighted term frequencies and assign to word columns
        article_count = len(in_range)
        feature_shit = row_outer.loc["vec_1":]
        print(f"in_range: {len(in_range)}, row: {feature_shit.shape}")
        # row_outer.loc["vec_1":] = [sum(x) / article_count for x in zip(*in_range)]  # with mean
        row_outer.loc["vec_1":] = [sum(x) for x in zip(*in_range)]  # just sum
        row_outer["article_count"] = article_count

    return row_outer


def create_doc2vec_cols_k_closest(row_outer, places_df, vector_size, max_dist):
    """Returns row along with mean of weighted doc2vec vector for articles closer than max_dist for property in row"""

    # in_range = []
    # for index, row in places_df.iterrows():
    #     dist = distance.distance(row["coords"], [row_outer["latitude"], row_outer["longitude"]]).m
    #     if dist < max_dist:
    #         # article is closer than max_dist m
    #         weight = 1 - (dist / max_dist)  # calculate weight between 0 and 1
    #         in_range.append(places_df.loc[index, "vec_1":] * weight)  # multiply tfs by weight

    dist_df = places_df.copy()
    dist_df["dist"] = dist_df.apply(
        lambda x: distance.distance(x["coords"], [row_outer["latitude"], row_outer["longitude"]]).m, axis=1)
    dist_df.sort_values(by=["dist"], inplace=True, ascending=True)  # sort by closest distance
    dist_df = dist_df.iloc[:10, :]

    # WEIGHTS?

    dist_df.drop(["dist"], axis=1, inplace=True)
    dist_df = dist_df.loc[:, "vec_1":]

    in_range = dist_df.to_numpy()

    # calculate mean of weighted term frequencies and assign to word columns
    article_count = dist_df.shape[0]
    # row_outer.loc["vec_1":] = [sum(x) / article_count for x in zip(*in_range)]  # with mean
    row_outer.loc["vec_1":] = [sum(x) for x in zip(*in_range)]  # just sum
    row_outer["article_count"] = article_count

    return row_outer


def add_doc2vec_features(input):
    """Multiprocessing service for process_text_features"""

    df_structured, places_df, max_dist = input
    vector_size = len([x for x in places_df.columns if "vec_" in x])

    # add column for amount of articles found
    row_count = df_structured.shape[0]
    df_structured["article_count"] = [0]*row_count

    for feature in ["vec_"+str(i) for i in range(1, vector_size+1)]:
        # add column for each word
        df_structured[feature] = [0]*row_count

    df_structured = df_structured.apply(create_doc2vec_cols, axis=1, result_type="broadcast",
                                        args=[places_df, vector_size, max_dist])
    # df_structured = df_structured.apply(create_doc2vec_cols_k_closest, axis=1, result_type="broadcast",
    #                                     args=[places_df, vector_size, max_dist])

    gc.collect()  # memory management
    return df_structured


def process_doc2vec_features(df_structured, places_df, max_dist, cores):
    """Returns structured df with added doc2vec features"""

    partitions = np.array_split(df_structured, 100)  # divide dataframe into 100 parts
    # add reference to places and tf data
    partitions = [(df, places_df, max_dist) for df in partitions]

    pool = Pool(processes=cores)
    results = []

    # Run partitions in parallel
    for x in tqdm(pool.imap_unordered(add_doc2vec_features, partitions), total=len(partitions)):
        results.append(x)

    pool.close()
    pool.join()

    return pd.concat(results, ignore_index=True)


# wrap execution in main method to guard multiprocessing code
if __name__ == "__main__":
    __spec__ = None  # remove spec to be able to run this script from iPython (Jupyter Notebook)
    compressed_path = sys.argv[1]
    cores = int(sys.argv[2])
    process(compressed_path, cores)

