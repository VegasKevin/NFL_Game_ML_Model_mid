"""Build one team-game row from nflverse play-by-play, using plays through Q3."""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
KEY = ["season", "game_id", "team"]
QKEY = KEY + ["qtr"]

# All source columns used below. Selecting them keeps one season manageable in memory.
COLUMNS = [
    "season", "game_id", "game_date", "season_type", "week", "play_id",
    "home_team", "away_team", "qtr", "quarter_seconds_remaining",
    "posteam", "defteam", "down", "ydstogo", "yardline_100", "fixed_drive",
    "home_timeouts_remaining", "away_timeouts_remaining", "play_type",
    "passing_yards", "rushing_yards", "sack", "interception",
    "first_down_rush", "first_down_pass", "first_down_penalty",
    "pass_touchdown", "rush_touchdown", "third_down_converted",
    "third_down_failed", "fumble_lost", "fumbled_1_team",
    "fumbled_2_team", "fumbled_2_player_id", "return_touchdown", "td_team",
    "extra_point_attempt", "two_point_attempt", "penalty", "penalty_team",
    "total_home_score", "total_away_score", "home_score", "away_score",
]

OFFENSE = {
    "sack": "sacks_allowed", "interception": "interceptions_thrown",
    "passing_yards": "passing_yards_gained", "rushing_yards": "rushing_yards_gained",
    "first_down_rush": "first_downs_rush_gained",
    "first_down_pass": "first_downs_pass_gained",
    "first_down_penalty": "first_downs_penalty_gained",
    "pass_touchdown": "pass_touchdowns_gained",
    "rush_touchdown": "rush_touchdowns_gained",
    "third_down_converted": "third_down_conversions",
    "third_down_failed": "third_down_failures",
}
DEFENSE = {
    "sack": "sacks_made", "interception": "interceptions_made",
    "passing_yards": "passing_yards_allowed", "rushing_yards": "rushing_yards_allowed",
    "first_down_rush": "first_downs_rush_allowed",
    "first_down_pass": "first_downs_pass_allowed",
    "first_down_penalty": "first_downs_penalty_allowed",
    "pass_touchdown": "pass_touchdowns_allowed",
    "rush_touchdown": "rush_touchdowns_allowed",
    "third_down_converted": "third_down_conversions_allowed",
    "third_down_failed": "third_down_stops",
}
HISTORY = {
    "win_credit": "pregame_win_pct",
    "passing_yards": "pregame_passing_yards_per_game",
    "rushing_yards": "pregame_rushing_yards_per_game",
    "points_for": "pregame_points_per_game",
    "points_against": "pregame_points_allowed_per_game",
    "scoring_margin": "pregame_avg_scoring_margin",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_season(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
    missing = set(COLUMNS) - set(frame.columns)
    require(not missing, f"Missing source columns: {sorted(missing)}")
    return frame


def download_season(season: int, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"play_by_play_{season}.csv"
    if target.exists() and target.stat().st_size:
        LOG.info("Using cached %s", target)
        return target
    url = f"{BASE_URL}/play_by_play_{season}.csv"
    part = target.with_suffix(".csv.part")
    LOG.info("Downloading %s", url)
    try:
        urllib.request.urlretrieve(url, part)
        require(part.stat().st_size > 0, f"Empty download: {url}")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return target


def count_events(frame: pd.DataFrame, team_col: str, fields: dict[str, str]) -> pd.DataFrame:
    source = frame.dropna(subset=[team_col])
    result = source.groupby(["season", "game_id", team_col, "qtr"], as_index=False)[list(fields)].sum()
    return result.rename(columns={team_col: "team", **fields})


def event_rows(frame: pd.DataFrame, team_col: str, name: str) -> pd.DataFrame:
    result = frame.dropna(subset=[team_col]).groupby(
        ["season", "game_id", team_col, "qtr"], as_index=False
    ).size()
    return result.rename(columns={team_col: "team", "size": name})


def game_metadata(pbp: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_date", "season_type", "week", "home_team", "away_team", "home_score", "away_score"]
    for col in cols:
        inconsistent = pbp.groupby(["season", "game_id"])[col].nunique(dropna=False).gt(1)
        require(not inconsistent.any(), f"Inconsistent {col} within a game")
    games = pbp.groupby(["season", "game_id"], as_index=False)[cols].first()
    require(games["home_team"].notna().all() and games["away_team"].notna().all(), "Missing game teams")
    require(games["home_team"].ne(games["away_team"]).all(), "Same home and away team")
    require(games[["home_score", "away_score"]].notna().all().all(), "Missing final scores")
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games


def quarter_scores(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    first = pbp[pbp.qtr.isin([1, 2, 3])].sort_values(["game_id", "play_id"]).copy()
    first[["total_home_score", "total_away_score"]] = first.groupby("game_id")[
        ["total_home_score", "total_away_score"]
    ].ffill()
    ends = first.groupby(["season", "game_id", "qtr"], as_index=False).tail(1)
    require(len(ends) == 3 * len(games), "A game lacks a Q1, Q2, or Q3 score row")
    require(ends[["total_home_score", "total_away_score"]].notna().all().all(), "Missing quarter-end score")
    ends = ends.sort_values(["game_id", "qtr"])
    for side in ("home", "away"):
        score = f"total_{side}_score"
        ends[f"{side}_points_scored"] = ends.groupby("game_id")[score].diff().fillna(ends[score])
        require(ends[f"{side}_points_scored"].ge(0).all(), "Quarter score decreased")
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        part = ends[["season", "game_id", "qtr", f"{side}_team",
                     f"total_{side}_score", f"total_{other}_score",
                     f"{side}_points_scored", f"{other}_points_scored"]].copy()
        part.columns = ["season", "game_id", "qtr", "team", "points_for",
                        "points_against", "points_scored", "opponent_points_scored"]
        part["point_diff"] = part.points_for - part.points_against
        part["point_diff_change"] = part.points_scored - part.opponent_points_scored
        part["combined_score"] = part.points_for + part.points_against
        part["absolute_margin"] = part.point_diff.abs()
        parts.append(part.drop(columns="opponent_points_scored"))
    return pd.concat(parts, ignore_index=True)


def quarter_events(pbp: pd.DataFrame) -> list[pd.DataFrame]:
    q = pbp[pbp.qtr.isin([1, 2, 3])].copy()
    tables = [count_events(q, "posteam", OFFENSE), count_events(q, "defteam", DEFENSE)]
    single = q[q.fumble_lost.eq(1) & q.fumbled_1_team.notna()
               & q.fumbled_2_team.isna() & q.fumbled_2_player_id.isna()]
    tables.append(event_rows(single, "fumbled_1_team", "fumbles_lost_single_fumble_plays"))
    tables.append(event_rows(q[q.return_touchdown.eq(1) & q.td_team.notna()],
                             "td_team", "return_touchdowns_scored"))
    rz = q[q.yardline_100.between(1, 20) & q.down.notna()
           & ~q.extra_point_attempt.eq(1) & ~q.two_point_attempt.eq(1)].copy()
    trips = rz.sort_values(["game_id", "play_id"]).drop_duplicates(
        ["season", "game_id", "posteam", "fixed_drive"]
    )
    tables.extend([event_rows(trips, "posteam", "red_zone_trips"),
                   event_rows(trips, "defteam", "red_zone_trips_allowed")])
    for team_col, suffix in (("posteam", ""), ("defteam", "_allowed")):
        fields = {"pass_touchdown": f"red_zone_passing_tds{suffix}",
                  "rush_touchdown": f"red_zone_rushing_tds{suffix}"}
        table = count_events(rz, team_col, fields)
        table[f"red_zone_touchdowns{suffix}"] = table[list(fields.values())].sum(axis=1)
        tables.append(table)
    tables.append(event_rows(q[q.penalty.eq(1)], "penalty_team", "penalty_plays"))
    return tables


def pregame_features(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        part = games[["season", "game_id", "game_date", f"{side}_team",
                      f"{side}_score", f"{other}_score"]].copy()
        part.columns = ["season", "game_id", "game_date", "team", "points_for", "points_against"]
        parts.append(part)
    team_games = pd.concat(parts, ignore_index=True)
    yards = pbp.dropna(subset=["posteam"]).groupby(
        ["season", "game_id", "posteam"], as_index=False
    )[["passing_yards", "rushing_yards"]].sum().rename(columns={"posteam": "team"})
    team_games = team_games.merge(yards, on=KEY, how="left", validate="one_to_one")
    team_games[["passing_yards", "rushing_yards"]] = team_games[
        ["passing_yards", "rushing_yards"]
    ].fillna(0)
    team_games["scoring_margin"] = team_games.points_for - team_games.points_against
    team_games["win_credit"] = team_games.scoring_margin.gt(0).astype(float) + 0.5 * team_games.scoring_margin.eq(0)
    team_games = team_games.sort_values(["season", "team", "game_date", "game_id"])
    grouped = team_games.groupby(["season", "team"])
    team_games["pregame_games_played"] = grouped.cumcount()
    for source, output in HISTORY.items():
        team_games[output] = grouped[source].transform(lambda x: x.shift(1).expanding().mean())
    team_games["won"] = team_games.scoring_margin.gt(0).astype(float).where(team_games.scoring_margin.ne(0))
    return team_games[KEY + ["pregame_games_played", *HISTORY.values(), "won"]]


def q3_snapshot(pbp: pd.DataFrame) -> pd.DataFrame:
    first_q4 = pbp[pbp.qtr.eq(4) & pbp.play_type.notna()].sort_values(
        ["game_id", "play_id"]
    ).groupby(["season", "game_id"], as_index=False).head(1).copy()
    cols = ["posteam", "down", "ydstogo", "yardline_100",
            "home_timeouts_remaining", "away_timeouts_remaining"]
    first_q4.loc[first_q4.quarter_seconds_remaining.ne(900), cols] = pd.NA
    return first_q4[["season", "game_id", "home_team", "away_team", *cols]].rename(columns={
        "posteam": "possession_team_q3", "down": "possession_down_q3",
        "ydstogo": "possession_yards_to_go_q3", "yardline_100": "possession_yards_to_goal_q3",
        "home_timeouts_remaining": "home_timeouts_q3", "away_timeouts_remaining": "away_timeouts_q3",
    })


def build_season(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    require(pbp.season.eq(season).all(), f"Source contains a season other than {season}")
    require(not pbp.duplicated(["game_id", "play_id"]).any(), "Duplicate game/play keys")
    games = game_metadata(pbp)
    quarterly = quarter_scores(pbp, games)
    for table in quarter_events(pbp):
        require(not table.duplicated(QKEY).any(), "Duplicate event aggregation keys")
        quarterly = quarterly.merge(table, on=QKEY, how="left", validate="one_to_one")
        new_cols = [c for c in table if c not in QKEY]
        quarterly[new_cols] = quarterly[new_cols].fillna(0)
    quarterly["third_down_attempts"] = quarterly.third_down_conversions + quarterly.third_down_failures
    quarterly["third_down_conversion_rate"] = quarterly.third_down_conversions.div(
        quarterly.third_down_attempts.replace(0, float("nan"))
    )
    quarterly["third_down_attempts_faced"] = (
        quarterly.third_down_conversions_allowed + quarterly.third_down_stops
    )
    quarterly["third_down_conversion_rate_allowed"] = quarterly.third_down_conversions_allowed.div(
        quarterly.third_down_attempts_faced.replace(0, float("nan"))
    )
    wide = quarterly.pivot(index=KEY, columns="qtr")
    wide.columns = [f"{name}_q{int(qtr)}" for name, qtr in wide.columns]
    result = wide.reset_index()
    history = pregame_features(pbp, games)
    result = result.merge(history, on=KEY, how="left", validate="one_to_one")
    snapshot = q3_snapshot(pbp)
    result = result.merge(snapshot, on=["season", "game_id"], how="left", validate="many_to_one")
    # Use game metadata even if a Q4 snapshot does not exist.
    home = games[["season", "game_id", "home_team", "away_team", "game_date", "season_type", "week"]]
    result = result.drop(columns=["home_team", "away_team"]).merge(
        home, on=["season", "game_id"], how="left", validate="many_to_one"
    )
    result["is_home"] = result.team.eq(result.home_team)
    # pandas 2.x cannot compare an object Series containing pd.NA directly.
    possession = result.possession_team_q3
    result["has_possession_q3"] = result.team.eq(possession.fillna("")).astype(float).where(
        possession.notna()
    )
    validate_output(result, games)
    return result.sort_values(["season", "game_date", "game_id", "team"]).reset_index(drop=True)


def validate_output(result: pd.DataFrame, games: pd.DataFrame) -> None:
    require(len(result) == 2 * len(games), "Expected exactly two rows per game")
    require(not result.duplicated(KEY).any(), "Duplicate team-game row")
    teams = result.groupby("game_id").team.agg(set)
    expected = games.set_index("game_id").apply(lambda r: {r.home_team, r.away_team}, axis=1)
    require(teams.eq(expected).all(), "Team-game rows do not match home/away teams")
    for qtr in (1, 2, 3):
        pairs = result.groupby("game_id")
        require(pairs[f"points_for_q{qtr}"].sum().eq(pairs[f"combined_score_q{qtr}"].first()).all(),
                f"Q{qtr} score symmetry failed")
        require(pairs[f"point_diff_q{qtr}"].sum().eq(0).all(), f"Q{qtr} margin symmetry failed")
    require(not {"opening_q4_play_type", "snapshot_clock", "play_type"}.intersection(result.columns),
            "Observed Q4 information entered output")


TOTAL_EVENTS = [
    "passing_yards_gained", "rushing_yards_gained",
    "passing_yards_allowed", "rushing_yards_allowed",
    "pass_touchdowns_gained", "rush_touchdowns_gained",
    "pass_touchdowns_allowed", "rush_touchdowns_allowed",
    "sacks_allowed", "sacks_made", "interceptions_thrown", "interceptions_made",
    "fumbles_lost_single_fumble_plays", "return_touchdowns_scored", "penalty_plays",
    "red_zone_trips", "red_zone_touchdowns",
    "red_zone_trips_allowed", "red_zone_touchdowns_allowed",
    "third_down_attempts", "third_down_conversions",
    "third_down_attempts_faced", "third_down_stops",
]


def consolidate_quarters(detailed: pd.DataFrame) -> pd.DataFrame:
    """Sum event columns within each team-game; leave pregame and snapshot fields intact."""
    metadata = ["season", "game_id", "team", "home_team", "away_team",
                "game_date", "season_type", "week", "is_home"]
    pregame = ["pregame_games_played", *HISTORY.values()]
    snapshot = ["possession_team_q3", "possession_down_q3",
                "possession_yards_to_go_q3", "possession_yards_to_goal_q3",
                "home_timeouts_q3", "away_timeouts_q3", "has_possession_q3"]
    result = detailed[metadata].copy()

    # Scoring retains its quarter-by-quarter view.
    for qtr in (1, 2, 3):
        result[f"points_scored_q{qtr}"] = detailed[f"points_scored_q{qtr}"]
        prior_score = detailed[f"points_against_q{qtr - 1}"] if qtr > 1 else 0
        result[f"points_allowed_q{qtr}"] = detailed[f"points_against_q{qtr}"] - prior_score
    result["score_diff_q3"] = detailed["point_diff_q3"]

    def total(name: str) -> pd.Series:
        return detailed[[f"{name}_q{qtr}" for qtr in (1, 2, 3)]].sum(axis=1)

    for name in TOTAL_EVENTS:
        result[f"{name}_through_q3"] = total(name)
    for side in ("gained", "allowed"):
        result[f"first_downs_{side}_through_q3"] = sum(
            total(f"first_downs_{kind}_{side}")
            for kind in ("rush", "pass", "penalty")
        )

    # Recalculate from total attempts. Zero attempts remains an undefined rate.
    attempts = result["third_down_attempts_through_q3"]
    result["third_down_conversion_rate_through_q3"] = (
        result["third_down_conversions_through_q3"] / attempts.replace(0, float("nan"))
    )
    faced = result["third_down_attempts_faced_through_q3"]
    result["third_down_conversion_rate_allowed_through_q3"] = (
        (faced - result["third_down_stops_through_q3"]) / faced.replace(0, float("nan"))
    )

    result[pregame + snapshot + ["won"]] = detailed[pregame + snapshot + ["won"]]
    require(len(result) == len(detailed), "Consolidation changed row count")
    require(not result.duplicated(KEY).any(), "Duplicate team-game after consolidation")
    for qtr in (1, 2, 3):
        pairs = result.groupby(["season", "game_id"])
        require(pairs[f"points_scored_q{qtr}"].sum().eq(
            pairs[f"points_allowed_q{qtr}"].sum()
        ).all(), f"Q{qtr} points are not symmetric")
    return result


def feature_dictionary(columns: list[str]) -> dict[str, str]:
    result = {}
    for name in columns:
        if name.startswith(("points_scored_q", "points_allowed_q")):
            result[name] = f"Points {name.split('_')[1]} in quarter {name[-1]}"
        elif name.endswith("_through_q3"):
            result[name] = f"{name.removesuffix('_through_q3').replace('_', ' ')}; total in Q1–Q3"
        elif name in HISTORY.values():
            source = next(k for k, v in HISTORY.items() if v == name)
            result[name] = f"Mean {source.replace('_', ' ')} from this team's earlier games in the same season"
        elif name == "won":
            result[name] = "Final-game label: 1 win, 0 loss, null tie; includes overtime"
        elif name == "score_diff_q3":
            result[name] = "Team score minus opponent score at the end of Q3"
        elif name.startswith("possession_") or name.endswith("_q3"):
            result[name] = "Pre-play Q4-opening situation, retained only when clock is 15:00"
        elif name == "pregame_games_played":
            result[name] = "Number of earlier games for this team in this season"
        else:
            result[name] = name.replace("_", " ")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2016, 2026)))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--csv", action="store_true", help="Also export a combined CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seasons = sorted(set(args.seasons))
    require(all(2016 <= year <= 2025 for year in seasons), "Seasons must be within 2016–2025")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for season in seasons:
        raw = download_season(season, args.raw_dir)
        result = build_season(read_season(raw), season)
        LOG.info("%s: %s games, %s team rows", season, result.game_id.nunique(), len(result))
        pieces.append(result)
    combined = consolidate_quarters(pd.concat(pieces, ignore_index=True))
    stem = f"team_games_q3_{seasons[0]}_{seasons[-1]}"
    combined.to_parquet(args.output_dir / f"{stem}.parquet", index=False)
    if args.csv:
        combined.to_csv(args.output_dir / f"{stem}.csv", index=False)
    (args.output_dir / f"{stem}_features.json").write_text(
        json.dumps(feature_dictionary(list(combined.columns)), indent=2) + "\n", encoding="utf-8"
    )
    LOG.info("Wrote %s rows to %s", len(combined), args.output_dir)


if __name__ == "__main__":
    main()
