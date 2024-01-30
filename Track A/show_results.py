import pandas as pd

lang = "eng"
data = pd.read_csv("pred_{}_a.csv".format(lang))

print(data.head(10))