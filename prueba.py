import pandas as pd
df=pd.read_csv("results.csv", sep=",")
costa_rica_tournaments=df[
    (df["home_team"]=="Costa Rica") |
    (df["away_team"]=="Costa Rica")
]

print(costa_rica_tournaments["tournament"].unique())
