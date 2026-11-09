"""Fill season-opening pregame averages from the preceding NFL season."""

from pathlib import Path

import pandas as pd

from pipeline import game_metadata, read_season


# Existing output column -> full-game team-game value to average.
METRICS = {
    "pregame_win_pct": "win_credit",
    "pregame_passing_yards_per_game": "passing_yards",
    "pregame_rushing_yards_per_game": "rushing_yards",
    "pregame_points_per_game": "points_for",
    "pregame_points_allowed_per_game": "points_against",
    "pregame_avg_scoring_margin": "scoring_margin",
}


def _team_averages(raw_path: Path) -> pd.DataFrame:
    """One row per team, averaging its previous season's completed games."""
    plays = read_season(raw_path)
    games = game_metadata(plays)

    home = games[["game_id", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["game_id", "team", "points_for", "points_against"]
    away = games[["game_id", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["game_id", "team", "points_for", "points_against"]
    team_games = pd.concat([home, away], ignore_index=True)

    yards = (
        plays.dropna(subset=["posteam"])
        .groupby(["game_id", "posteam"], as_index=False)[
            ["passing_yards", "rushing_yards"]
        ]
        .sum()
        .rename(columns={"posteam": "team"})
    )
    team_games = team_games.merge(
        yards, on=["game_id", "team"], how="left", validate="one_to_one"
    )
    team_games[["passing_yards", "rushing_yards"]] = team_games[
        ["passing_yards", "rushing_yards"]
    ].fillna(0)
    team_games["scoring_margin"] = team_games.points_for - team_games.points_against
    team_games["win_credit"] = (
        team_games.scoring_margin.gt(0).astype(float)
        + 0.5 * team_games.scoring_margin.eq(0)
    )

    return team_games.groupby("team", as_index=False).agg(
        **{f"prior_{column}": (source, "mean") for column, source in METRICS.items()}
    )


def fill_first_game_from_previous_season(
    frame: pd.DataFrame, raw_dir: Path
) -> pd.DataFrame:
    """Fill missing opener values in the six existing pregame columns.

    The first season remains missing when its predecessor is not cached.
    Genuine zero values and non-opener rows are left as they are.
    """
    result = frame.copy()
    seasons = sorted(result.season.unique())
    summaries = []
    for season in seasons:
        raw_path = Path(raw_dir) / f"play_by_play_{int(season) - 1}.csv"
        if not raw_path.exists():
            if season == seasons[0]:
                continue
            raise FileNotFoundError(f"Previous-season data is missing: {raw_path}")
        summary = _team_averages(raw_path)
        summary["season"] = season
        summaries.append(summary)

    if not summaries:
        return result

    previous = pd.concat(summaries, ignore_index=True)
    result = result.merge(
        previous, on=["season", "team"], how="left", validate="many_to_one"
    )
    opener = result.pregame_games_played.eq(0)
    for column in METRICS:
        missing = opener & result[column].isna()
        result.loc[missing, column] = result.loc[missing, f"prior_{column}"]
    return result.drop(columns=[f"prior_{column}" for column in METRICS])
