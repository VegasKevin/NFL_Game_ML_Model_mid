from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


import seaborn as sns
from matplotlib import pyplot as plt


from previous_season import fill_first_game_from_previous_season

#File Paths for full Data Set
dataset_dir = Path(__file__).resolve().parent
dataset_path = dataset_dir / "data" / "processed" / "team_games_q3_2016_2025.parquet"


full_set_df = pd.read_parquet(dataset_path)

print("Full Data Set Shape of DF: ", full_set_df.shape)

#standardize column names
full_set_df.columns = full_set_df.columns.str.lower().str.replace(' ', '_')


#Check for missing values
missing_values = full_set_df.isnull().sum()
print("Missing Values before Cleaning:")
print(missing_values[missing_values > 0])

#Removing Ties from Data Set, as they are not a win or loss and will not be used in the model
full_set_df = full_set_df.dropna(subset=["won"]).copy()


# Fill missing opener values using the previous season's completed games.
# We use a value of average points, years and other values through the season, but we don't have it for the first game of the season.
# We fill those values for this game of the year using Last Year's Values.  First season will still have zeros because there is no previous season to pull from.
full_set_df = fill_first_game_from_previous_season(
    full_set_df, dataset_dir / "data" / "raw"
)


#print(full_set_df[full_set_df["pregame_win_pct"] == 0])

# Its common for a Quarter to not have a third down conversion, we fill these with Zeros instead of NaN
rate_cols = [
    col for col in full_set_df.columns
    if col.startswith("third_down_conversion_rate")
]
full_set_df[rate_cols] = full_set_df[rate_cols].fillna(0)


#Decided to leave the remaining NaN values because Zero for some of these can indicate something else and there are few that it shouldn't impact the data set.
missing_values = full_set_df.isnull().sum()
print("Missing Values after Cleaning:")
print(missing_values[missing_values > 0])




#Give Post season Object type value Numeric Value
is_post_season = {"REG": 0, "POST": 1}
full_set_df["season_type"] = full_set_df["season_type"].map(is_post_season)

#print(full_set_df.select_dtypes(include= "int64").head(30))
#print(full_set_df.dtypes)

#Define Relevant Columns for Model
numerical_columns = full_set_df.select_dtypes(include=["int64","float64"]).columns.tolist()
numerical_columns.remove("season")
numerical_columns.remove("week")  



### START TRAINING DATA ##

#Split Training, test and Val data based on Game IDs

game_ids = full_set_df["game_id"].unique()

train_val_games, test_games = train_test_split(
    game_ids, test_size=0.2, random_state=42
)
train_games, val_games = train_test_split(
    train_val_games, test_size=0.25, random_state=42
)

df_train = full_set_df.loc[
    full_set_df["game_id"].isin(train_games), numerical_columns
].reset_index(drop=True)

df_val = full_set_df.loc[
    full_set_df["game_id"].isin(val_games), numerical_columns
].reset_index(drop=True)

df_test = full_set_df.loc[
    full_set_df["game_id"].isin(test_games), numerical_columns
].reset_index(drop=True)

'''#Create Train, Test and Validation Sets with Train_Test_Split
df_full_train, df_test = train_test_split(full_set_df[numerical_columns], test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
#Reset Indexes
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
#Determine True Values'''
y_train = (df_train["won"] == 1).astype(int)
y_val = (df_val["won"] == 1).astype(int)
y_test = (df_test["won"] == 1).astype(int)  
#Remove Y Column from Data Sets
del df_train["won"]
del df_val["won"]
del df_test["won"]

#print(df_train)
train_dicts = df_train.to_dict(orient="records")

## DECISION TREE Model
dv = DictVectorizer(sparse=False)
X_Train = dv.fit_transform(train_dicts)


tree_scores = []

for depth in [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, None]:
    for s in [ 2, 5, 10, 15, 20, 25, 30, 50, 100, 500]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_split=s, random_state=42)
        dt.fit(X_Train, y_train)
        val_dicts = df_val.to_dict(orient="records")
        X_val = dv.transform(val_dicts)
        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        tree_scores.append((depth, s, auc))
        #print(f"Depth: {depth}, Min Samples Split: {s}, AUC: {auc:.4f}")

print("Top 10 based on AUC score")

columns = ["max_depth", "min_samples_split", "auc"]
df_tree_scores = pd.DataFrame(tree_scores, columns=columns)
print(df_tree_scores.nlargest(10, "auc").to_string(index=False))
df_scores_pivot = df_tree_scores.pivot(index='min_samples_split', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)
sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")
plt.savefig("data/processed/tree_auc_heatmeap.png", dpi=150)
plt.close()
print("Saved heatmap to data/processed/tree_auc_heatmap.png")

#Choosing Values of Max_depth = 4 and Min_sample_split = 500;  It gives a good AUC with only Depth of 4
print("Choosing Values of Max_depth = 4 and Min_sample_split = 500;  It gives a AUC of .8637 with only Depth of 4")

final_Dec_tree = DecisionTreeClassifier(max_depth=4, min_samples_split=500)
final_Dec_tree.fit(X_Train, y_train)

print(export_text(final_Dec_tree, feature_names=list(dv.get_feature_names_out())))


#Logistic Regression
lr_X_train = df_train.fillna(0)
lr_X_val = df_val.fillna(0)

model = LogisticRegression ()
model.fit(lr_X_train, y_train)
lr_y_pred = model.predict_proba(lr_X_val)[:,1]
print("Logistic Regression AUC: ", round(roc_auc_score(y_val, lr_y_pred),4))









