"""Small hand-checked games test the feature boundary and joins."""

import unittest

import pandas as pd

from pipeline import COLUMNS, build_season


def play(game, date, home, away, home_final, away_final, qtr, play_id,
         home_running, away_running, **changes):
    row = {column: 0 for column in COLUMNS}
    row.update({
        "season": 2025, "game_id": game, "game_date": date,
        "season_type": "REG", "week": 1 if game == "g1" else 2,
        "home_team": home, "away_team": away,
        "home_score": home_final, "away_score": away_final,
        "qtr": qtr, "play_id": play_id,
        "quarter_seconds_remaining": 900 if qtr == 4 else 0,
        "posteam": home, "defteam": away, "play_type": "pass",
        "total_home_score": home_running, "total_away_score": away_running,
        "fixed_drive": play_id, "down": 1, "ydstogo": 10, "yardline_100": 75,
        "home_timeouts_remaining": 3, "away_timeouts_remaining": 3,
        "fumbled_1_team": None, "fumbled_2_team": None,
        "fumbled_2_player_id": None, "td_team": None, "penalty_team": None,
    })
    row.update(changes)
    return row


class PipelineTest(unittest.TestCase):
    def test_two_games_and_prior_game_only(self):
        rows = [
            play("g1", "2025-09-01", "A", "B", 21, 14, 1, 1, 7, 0,
                 passing_yards=50, pass_touchdown=1, yardline_100=15,
                 third_down_converted=1),
            play("g1", "2025-09-01", "A", "B", 21, 14, 2, 2, 14, 7,
                 rushing_yards=20, rush_touchdown=1, yardline_100=10,
                 third_down_failed=1),
            play("g1", "2025-09-01", "A", "B", 21, 14, 3, 3, 14, 14,
                 passing_yards=30, penalty=1, penalty_team="B",
                 return_touchdown=1, td_team="B", fumble_lost=1,
                 fumbled_1_team="A"),
            play("g1", "2025-09-01", "A", "B", 21, 14, 4, 4, 14, 14,
                 passing_yards=999, down=2, ydstogo=5, yardline_100=40),
            play("g2", "2025-09-08", "A", "C", 14, 17, 1, 1, 3, 0,
                 passing_yards=10),
            play("g2", "2025-09-08", "A", "C", 14, 17, 2, 2, 3, 7),
            play("g2", "2025-09-08", "A", "C", 14, 17, 3, 3, 10, 7),
            play("g2", "2025-09-08", "A", "C", 14, 17, 4, 4, 10, 7),
        ]
        result = build_season(pd.DataFrame(rows), 2025)
        self.assertEqual(len(result), 4)
        a1 = result.set_index(["game_id", "team"]).loc[("g1", "A")]
        a2 = result.set_index(["game_id", "team"]).loc[("g2", "A")]
        b1 = result.set_index(["game_id", "team"]).loc[("g1", "B")]
        self.assertEqual(a1.passing_yards_gained_q1, 50)
        self.assertEqual(a1.points_for_q3, 14)
        self.assertEqual(a1.red_zone_trips_q1, 1)
        self.assertEqual(a1.fumbles_lost_single_fumble_plays_q3, 1)
        self.assertEqual(b1.return_touchdowns_scored_q3, 1)
        self.assertEqual(b1.penalty_plays_q3, 1)
        self.assertEqual(b1.points_for_q3, a1.points_against_q3)
        self.assertEqual(a1.third_down_conversion_rate_q1, 1)
        self.assertTrue(pd.isna(a1.third_down_conversion_rate_q3))
        self.assertTrue(pd.isna(a1.pregame_win_pct))
        self.assertEqual(a1.possession_down_q3, 2)
        self.assertEqual(a2.pregame_win_pct, 1)
        self.assertEqual(a2.pregame_passing_yards_per_game, 1079)
        self.assertEqual(a2.pregame_points_per_game, 21)
        self.assertEqual(a2.won, 0)
        self.assertNotIn("play_type", result.columns)
        self.assertNotIn("snapshot_clock", result.columns)

    def test_late_q4_opening_snapshot_is_null(self):
        rows = [
            play("g1", "2025-09-01", "A", "B", 7, 0, qtr, qtr, 0, 0)
            for qtr in (1, 2, 3)
        ]
        rows.append(play("g1", "2025-09-01", "A", "B", 7, 0, 4, 4, 0, 0,
                         quarter_seconds_remaining=880))
        result = build_season(pd.DataFrame(rows), 2025)
        self.assertTrue(result.possession_down_q3.isna().all())
        self.assertTrue(result.has_possession_q3.isna().all())

    def test_final_tie_has_no_label(self):
        rows = [
            play("g1", "2025-09-01", "A", "B", 10, 10, qtr, qtr, 0, 0)
            for qtr in (1, 2, 3)
        ]
        rows.append(play("g1", "2025-09-01", "A", "B", 10, 10, 4, 4, 0, 0))
        result = build_season(pd.DataFrame(rows), 2025)
        self.assertTrue(result.won.isna().all())


if __name__ == "__main__":
    unittest.main()
