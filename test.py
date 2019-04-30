import pandas as pd

df = pd.read_csv("data/AdmissionDataset/data.csv")

df["Chance of Admit"] = [1 if chance >= 0.5 else 0 for chance in df['Chance of Admit']]

print(df.head(20))

df.to_csv("updated.csv", index=False)
