import pandas as pd
import sys


table = pd.read_csv(sys.argv[1])
table["datacloud"] = table.name.apply(lambda row: row.split("__")[0])
table["datasegment"] = table.datacloud.apply(lambda row: row.split("-")[0])
idx = table.groupby(['datacloud'])['score'].transform(min) == table['score']

table = table.loc[idx]
table[["name", "datasegment"]].to_csv("selected.csv", index=False)
