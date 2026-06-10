import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import time
#============================================================================================
#This is an example of the matches from Costa Rica in soccer from 1921 to 2025
#We gonna include fendly matches and Turnament matches, the stadistics of wins and losses
#and the most important matches in the history of Costa Rica soccer
#============================================================================================

#Load the dataset

print("\n loading dataset....")

df=pd.read_csv("jalanza2002/ML-for-Costa-Rica-soccer-team-/results.csv", sep=",")

print("\n print the matches from costa rica....")
print(df[df["home_team"]=="Costa Rica"])
print("------------------------------------------")
print(df[df["away_team"]=="Costa Rica"])
#=======================================================================

print("\n print the information of the dataset from Costa Rica....")
print(df[df["home_team"]=="Costa Rica"].info())
#=======================================================================

columns, rows = df[df["home_team"]=="Costa Rica"].shape
print(f"the dataset has {columns} columns and {rows} rows")
#=======================================================================

print("\n print the description of the dataset from Costa Rica....")
print(df[df["home_team"]=="Costa Rica"].describe())
#=======================================================================

print("\n print the nulls from Costa Rica dataset....")
total_nulls = df[df["home_team"]=="Costa Rica"].isnull().sum().sum()
print(f"Total null values in the Costa Rica dataset: {total_nulls}")
print("")
print(df[df["home_team"]=="Costa Rica"].isnull().sum())
#=======================================================================

print("\n Mean and median of the goals scored by Costa Rica in home matches....")
print(f"Mean goals: {df[df['home_team'] == 'Costa Rica']['home_score'].mean()}")
print(f"Median goals: {df[df['home_team'] == 'Costa Rica']['home_score'].median()}")
#=======================================================================

print("\n Mean and median of goals scored by Costa Rica in away matches....")
print(f"Mean goals: {df[df['away_team'] == 'Costa Rica']['away_score'].mean()}")
print(f"Median goals: {df[df['away_team'] == 'Costa Rica']['away_score'].median()}")
#=======================================================================

print("\n desviacion estandar of goals scored by Costa Rica in home matches and away matches....")
print(f"Standard deviation goals home matches: {df[df['home_team'] == 'Costa Rica']['home_score'].std()}")
print(f"Standard deviation goals away matches: {df[df['away_team'] == 'Costa Rica']['away_score'].std()}")
#=======================================================================

print("\n print the corrlation between the goals scored by Costa Rica" \
"in home matches and away matches during 1921 to 2022....")

df_corr= df.copy()

le_home = LabelEncoder()
le_away = LabelEncoder()
le_date = LabelEncoder()

df_corr["home_team"] = le_home.fit_transform(df_corr["home_team"]=='Costa Rica')
df_corr["away_team"] = le_away.fit_transform(df_corr["away_team"]=='Costa Rica')
print("\n the corelation from the goals scored by Costa Rica in home matches and away matches during 1921 to 2022....")
corelation= df_corr[["home_team", "away_team", "home_score", "away_score"]].corr()
print(corelation)
#=======================================================================

print("\n print a pie chart of wins and loses of Costa Rica during 1921 to 2025....")

costa_rica_matches = df[(df["home_team"] == "Costa Rica") | 
                        (df["away_team"] == "Costa Rica") 
                        ]

win=0
lose=0
draws=0

for _, row in costa_rica_matches.iterrows():
    if row["home_team"] == "Costa Rica":# this count the wins and loses of Costa Rica in home matches
        if row["home_score"] > row["away_score"]:
            win += 1
        elif row["home_score"] < row["away_score"]:
            lose += 1
        else:
            draws += 1
    else:#this count the wins and loses of Costa Rica in away matches
        if row["away_score"] > row["home_score"]:
            win += 1
        elif row["away_score"] < row["home_score"]:
            lose += 1
        else:
            draws += 1

labels = ["Wins", "Losess", "Draws"]
values = [win, lose, draws]

plt.figure(figsize=(6, 6))
plt.pie(values, 
        labels=labels, 
        autopct="%1.1f%%", 
        startangle=90, 
        colors=["#4CAF50", "#F44336", "#B8B8B8"]
        )
print("\n the pie chart of wins and loses of Costa Rica during 1921 to 2022....")
plt.title("Wins and Loses of Costa Rica (1921-2022)")
plt.show()
#=======================================================================
time.sleep(2)
print("\n print a bar chart of the most years that Costa Rica scored more goals in home matches during 1921 to 2022....")

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

# Scores for eac match for Costa Rica
df["cr_goals"] = 0

df.loc[df["home_team"] == "Costa Rica", "cr_goals"] = df["home_score"]
df.loc[df["away_team"] == "Costa Rica", "cr_goals"] = df["away_score"]

# Just matches where Costa Rica played
costa_rica_matches = df[
    (df["home_team"] == "Costa Rica") |
    (df["away_team"] == "Costa Rica")
]

# Do a range of years to group the matches, for example: 1920-1925, 1925-1930, etc.
bins = range(1920, 2025, 5)

costa_rica_matches["year_group"] = pd.cut(
    costa_rica_matches["year"],
    bins=bins
)

# Sum the scores of Costa Rica by year group
goals_by_period = (
    costa_rica_matches
    .groupby("year_group")["cr_goals"]
    .sum()
)
plt.figure(figsize=(10, 6))

plt.bar(
    goals_by_period.index.astype(str),
    goals_by_period.values,
    color="#4CAF50"
)

plt.title("Goals Scored by Costa Rica (1921-2025)")
plt.xlabel("5-Year Period")
plt.ylabel("Goals Scored")

for i, value in enumerate(goals_by_period.values):
    plt.text(i, value + 2, str(value), ha="center")

plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()
#=======================================================================
print("\n print in wich tourment Costa Rica score more")

costa_rica_tournaments=df[
    (df["home_team"]=="Costa Rica") |
    (df["away_team"]=="Costa Rica")
]

goals_by_tourment={}

for _, row in costa_rica_tournaments.iterrows():

    tournament= row["tournament"]

    if row["home_team"]=="Costa Rica":
        goals= row["home_score"]
    else:
        goals= row["away_score"]
    
    if row["away_team"]=="Costa Rica":
        goals = row["away_score"]
    else:
        goals = row["home_score"]

    if tournament not in goals_by_tourment:
        goals_by_tourment[tournament]= goals
    else:
        goals_by_tourment[tournament] +=goals




plt.figure(figsize=(12,8))

plt.barh(
    list(goals_by_tourment.keys()),
    list(goals_by_tourment.values())
)
plt.title("Goals Scored by Costa Rica by Tournament")
plt.xlabel("Goals")
plt.ylabel("Tournament")

plt.tight_layout()
plt.show()

#=======================================================================
#Now here we goint to use the scalte the data
#=======================================================================

scaler= StandardScaler()

df_ml=df.copy()

# Filas antes de la limpieza
rows_before = df_ml.shape[0]

# Eliminar todas las filas que contengan 'unknown'
df_ml = df_ml[~df_ml.isin(['unknown']).any(axis=1)]

# Filas después de la limpieza
rows_after = df_ml.shape[0]

print(f"We deleted {rows_before - rows_after} with the data 'unknown'.")

df_cr= df_ml[(df_ml["home_team"]=="Costa Rica")|
             (df_ml["away_team"]=="Costa Rica")]

numerics_columns=["away_score","home_score"]

df_cr[numerics_columns]= scaler.fit_transform(
    df_cr[numerics_columns]
)
print("the escalade are finished")
print("\n the first columns for the escalade")
print(df[numerics_columns].head())
#==========================================================================================================
print("\n print the description of the escalade dataframe")
print(df[numerics_columns].describe())

#===========================================================================================================
#Now we going to predic if COsta Rica can score more than two goals when they are playing outside of they home
#for this we going to use XG Boost and the decision tree to can predict this information
#=============================================================================================================

def goals_from_CR(row):

    if row["home_team"]=="Costa Rica":
        return row["home_score"]
    else:
        return row["away_score"]
    
df_cr["goals_cr"]= df_cr.apply(goals_from_CR, axis=1)
df_cr["high_scoring"]= (df_cr["goals_cr"]>=2).astype(int)

def opponent(row):
    if row["home_team"]=="Costa Rica":
        return row["away_team"]
    else:
        return row["home_team"]
    

df_cr["opponent"]= df_cr.apply(opponent, axis=1)

top_opponents = df_cr["opponent"].value_counts().head(5).index
top_tournaments = df_cr["tournament"].value_counts().head(5).index

# Filtrate the data
df_cr = df_cr[
    (df_cr["opponent"].isin(top_opponents)) &
    (df_cr["tournament"].isin(top_tournaments))
].copy()

df_pred = df_cr[["opponent","tournament", "neutral", "year", "high_scoring"]].copy()

df_pred = pd.get_dummies(
    df_pred,
    columns=["opponent","tournament"],
    drop_first=True
)

X = df_pred.drop("high_scoring", axis=1)
y = df_pred["high_scoring"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model= RandomForestClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train,y_train)

print("\n Random Forest Tree already trained")

y_pred= model.predict(X_test)

accurancy= round(accuracy_score(y_test, y_pred), 4)

print("\n The accurancy report is: ", accurancy)

print("The report for the random forest is:")
print(classification_report(y_test, y_pred))

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(12,6))
plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Feature Importance - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#===============================================
#K-means
#===============================================

df_kmeans= df_cr.copy()

variables=["opponent","tournament", "high_scoring"]
x=df_kmeans[variables]

#80/20
X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# find the best K
scores = []

for k in range(2, 5):
    modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_temp = modelo_temp.fit_predict(X_train_scaled)
    score = silhouette_score(X_train_scaled, clusters_temp)
    scores.append((k, score))
best_k= max(scores, key=lambda x: x[1])[0]
print(f"Mejor número de clusters: {best_k}")

# final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_kmeans["cluster"] = kmeans.fit_predict(scaler.transform(X))

print("Kmeans already finished.")
print(df_kmeans["cluster"].value_counts())

# Graphic
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df_kmeans,
    x="opponent",
    y="tournament",
    hue="cluster",
    palette="viridis",
    s=70,
    alpha=0.75
)

plt.title("Kmeans for Costa Rica matches", fontsize=16, fontweight="bold")
plt.xlabel("opponent")
plt.ylabel("tournament")
plt.grid(True, alpha=0.3)
plt.legend(title="Cluster")

plt.show()

#===============================================================================
#XGBoost to predict if Costa Rica Score one goal depends if is home or away match
#=================================================================================
print("\n-----------------XGBOOST for the top 5-----------------")

df_xg = df_kmeans.copy()

df_xg["goals_for"]=np.where(
    df_xg["home_team"]=="Costa Rica",
    df_xg["home_score"],
    df_xg["away_score"]
)

df_xg["scores_goal"]= (df_xg["goals_for"]>0).astype(int)

# Select variables
df_xg = df_xg[
    ["opponent", "tournament", "cluster", "scores_goal"]
]

# Encoding the variables
df_xg = pd.get_dummies(
    df_xg,
    columns=["opponent", "tournament"],
    drop_first=True
)

# predict variables 
X = df_xg.drop("scores_goal", axis=1)
y = df_xg["scores_goal"]

# Divide train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Model XGBoost
model_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42
)

# train
model_xgb.fit(X_train, y_train)

# Predictions
y_pred = model_xgb.predict(X_test)

# Metrics
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="magma",
    linewidths=2,
    linecolor="white",
    xticklabels=["No High Scoring", "High Scoring"],
    yticklabels=["No High Scoring", "High Scoring"]
)

plt.title("XGBoost - Matriz de Confusión", fontsize=16, fontweight="bold")
plt.xlabel("Prediction")
plt.ylabel("Real Value")

plt.show()