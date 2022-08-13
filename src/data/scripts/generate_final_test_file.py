import pandas as pd

df = pd.read_csv("../../../data/source_files/liverpool_vs_watford_random_batch_fully_tagged")
df = df.drop("label_promoter", axis=1)
df.columns = ["original_index", *df.columns[1:]]
df.to_csv("liverpool_vs_watford_random_batch_fully_tagged", index=False)
