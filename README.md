# Wikipedia Real Estate

## Downloading data and creating ML features
To download and preprocess structured housing data, the notebook
``download_basic.ipynb`` needs to be run. To download data about asset types,
used for GIS based features, ``download_gis.ipynb`` needs to be run. To download
the complete English Wikipedia __(17 GB of disk space needed!)__ and select
articles from Allegheny county, ``download_wikipedia.ipynb`` needs to be run. To
preprocess data from Wikipedia and to create ML features,
``process_wikipedia.ipynb`` needs to be run.


## Training and evaluating models
In order to create and train models, the corresponding ``download_`` notebook
needs to be run __first__. After that, different kind of models are trained,
tested and validated by running the respective ``ML_`` notebook. Results are shown
in the notebook and saved in the "results" folder. Note that if a different
``COUNT_RADIUS`` or ``MAX_DIST`` should be evaluated, the data sets with the
desired ``COUNT_RADIUS`` or ``MAX_DIST`` need to be created first (see above).

## Visualizing data and results
Data and results can be visualized by running the appropriate ``visualize_``
notebook. A mapbox token is needed for some visualizations.

## Requirements
Below is a list of necessary packages along with versions which were used in
development. Dependencies of used packages are not necessarily listed.

| package          | version       |
| ---------------- | ------------- |
| keras            | 2.4.3         |
| pandas           | 1.1.3         |
| numpy            | 1.19.2        |
| matplotlib       | 3.3.2         |
| seaborn          | 0.11.1        |
| plotly           | 4.14.1        |
| tqdm             | 4.54.1        |
| beautifulsoup4   | 4.9.3         |
| requests         | 2.25.0        |
| xml-python       | 0.3.5         |
| mwparserfromhell | 0.5.4         |
| geopy            | 2.1.0         |
| catboost         | 0.24.4        |
| xgboost          | 1.3.2         |
| scikit-learn     | 0.23.2        |
| nltk             | 3.5           |
