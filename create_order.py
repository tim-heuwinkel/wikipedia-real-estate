PATH = "C:/Users/Tim/.keras/datasets/wikipedia_real_estate/"
MAX_DIST = 6500
K_FOLDS = 5
WEIGHTING = True
MEAN = False
DUMMIES = ["MUNICODE"]  # e.g. ["MUNICODE"]

meaned = pd.read_csv(PATH + f"structured_wiki_text_features_{MAX_DIST}.csv")
non_meaned = pd.read_csv(PATH + f"structured_wiki_text_features_{MAX_DIST}_NOMEAN.csv")

rows = []
col_names = list(non_meaned.columns)
for index, row in meaned.iterrows():
    non_meaned_row = non_meaned[non_meaned["_id"]==row["_id"]]
    rows.append(non_meaned_row.values[0])
non_meaned = pd.DataFrame(rows, columns=col_names)
non_meaned.head(10)

non_meaned.to_csv(PATH + f"structured_wiki_text_features_{MAX_DIST}_NOMEAN.csv", index=False)
