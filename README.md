# NFL win probability at the end of Q3

This project builds a team-game dataset for predicting whether a team wins after the third quarter. Each game has two rows, one per team. The `won` label uses the final result, including overtime; tied games have no label.

The default data covers NFL seasons **2016–2025**, regular season and playoffs. Games in calendar year 2026 count when they belong to the 2025 season. Predictors use play data through Q3, earlier games for pregame averages, and pre-play situation fields from the opening Q4 record when its clock is 15:00. No outcome from that Q4 play is used.

## Run from a GitHub clone

Install Git and Docker Desktop, and start Docker. Then open a terminal:

```sh
git clone <GitHub-repository-URL>
cd NFL_Game_ML_Model
docker compose build
docker compose run --rm pipeline
```

Replace the placeholder with this repository's GitHub URL after it is published. The first run downloads nflverse play-by-play files; later runs reuse the local cache. The container exits after writing the dataset.

For a quicker trial of one season:

```sh
docker compose run --rm pipeline pipeline.py --seasons 2024
```

To export CSV as well as Parquet:

```sh
docker compose run --rm pipeline pipeline.py --seasons 2024 --csv
```

## Files created

- `data/raw/`: downloaded season files, cached on your computer.
- `data/processed/team_games_q3_2016_2025.parquet`: default team-game dataset.
- `data/processed/team_games_q3_2016_2025_features.json`: feature dictionary.
- `data/processed/*.csv`: optional CSV exports.

The two data folders are mounted from the host, so files survive after the container exits. They are excluded from Git. The source release can be revised; retain cached raw files if you need to reproduce an exact previous build.

## Model exploration

`Clean_Data_1.py` is a separate, exploratory cleaning and modeling script. It is not part of the dataset-building command. After editing it, rebuild the image, then run:

```sh
docker compose build
docker compose run --rm pipeline Clean_Data_1.py
```

Before relying on this model run, correct its season-type mapping to use `{"REG": 0, "POST": 1}` and have its logistic-regression prediction use `lr_X_val`. The script is still being developed; its AUC is not a published benchmark. Train, validation, and test splits should keep both team rows from a game together.

## Dataset notes

The pipeline keeps points scored and allowed by quarter; the selected play events are totals through Q3. Pregame averages use completed earlier games in the same season. First-game pregame averages are missing in the raw pipeline output; the separate cleaning script handles its own fill. Undefined third-down rates and unavailable Q3 situation measurements remain missing in the pipeline output. The original `Data_Import.ipynb` is for exploration and is not run by Docker.

Source: [nflverse play-by-play releases](https://github.com/nflverse/nflverse-data/releases/tag/pbp).
