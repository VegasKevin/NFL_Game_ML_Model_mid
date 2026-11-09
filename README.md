# NFL win probability at the end of Q3

This project uses [nflverse play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp) to build a dataset for predicting a team's chance of winning at the end of the third quarter. It has two rows per game, one for each team. Final results, including overtime, provide the win/loss label.

## Clone and run

Install Git and Docker Desktop, and start Docker. Then run:

```sh
git clone https://github.com/VegasKevin/NFL_Game_ML_Model_mid.git
cd NFL_Game_ML_Model_mid
docker compose build
docker compose run --rm pipeline
docker compose run --rm pipeline Clean_Data_1.py
```

The first run downloads and caches the 2016–2025 seasons and writes the dataset. The second runs the exploratory cleaning and model comparisons, prints their scores, and saves a heatmap. Both regular season and playoff games are included. The container exits when each command finishes.

## What is where

- `pipeline.py` builds the Q3 features and labels. Its output is `data/processed/team_games_q3_2016_2025.parquet`, with a JSON feature dictionary alongside it.
- `Clean_Data_1.py` fills selected missing values and compares decision tree and logistic regression models. `previous_season.py` supports its prior-season fills.
- `Data_Import.ipynb` is the original exploration notebook; Docker does not run or change it.
- `data/raw/` holds cached downloads; `data/processed/` holds generated files. Docker mounts both folders from your computer, and Git excludes them.

Model scores are exploratory validation results, not a deployed prediction service. Keep both team rows from a game in the same train, validation, or test split.
