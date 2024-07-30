import pandas as pd
import pytest

from attribution.experiment_logger import ExperimentLogger


@pytest.fixture
def experiment_logger():
    return ExperimentLogger()


@pytest.mark.parametrize(
    "input_df, score_agg, expected_df",
    [
        # Test case 1: Aggregation using mean, with output_token_pos
        (
            pd.DataFrame(
                {
                    "input_token_pos": [0, 0, 1, 1],
                    "output_token_pos": [0, 0, 1, 1],
                    "attr_score": [1, 2, 3, 4],
                    "other_col": ["a", "b", "c", "d"],
                }
            ),
            "mean",
            pd.DataFrame(
                {
                    "input_token_pos": [0, 1],
                    "output_token_pos": [0, 1],
                    "attr_score": [1.5, 3.5],
                    "other_col": ["b", "d"],
                }
            ),
        ),
        # Test case 2: Aggregation using sum, without output_token_pos
        (
            pd.DataFrame(
                {
                    "input_token_pos": [0, 0, 1, 1],
                    "attr_score": [1, 2, 3, 4],
                    "other_col": ["a", "b", "c", "d"],
                }
            ),
            "sum",
            pd.DataFrame(
                {"input_token_pos": [0, 1], "attr_score": [3, 7], "other_col": ["b", "d"]}
            ),
        ),
        # Test case 3: Aggregation using last, with output_token_pos
        (
            pd.DataFrame(
                {
                    "input_token_pos": [0, 0, 1, 1],
                    "output_token_pos": [0, 0, 1, 1],
                    "attr_score": [1, 2, 3, 4],
                    "other_col": ["a", "b", "c", "d"],
                }
            ),
            "last",
            pd.DataFrame(
                {
                    "input_token_pos": [0, 1],
                    "output_token_pos": [0, 1],
                    "attr_score": [2, 4],
                    "other_col": ["b", "d"],
                }
            ),
        ),
        # Test case 4: Aggregation using mean, without output_token_pos
        (
            pd.DataFrame(
                {
                    "input_token_pos": [0, 0, 1, 1],
                    "attr_score": [1, 2, 3, 4],
                    "other_col": ["a", "b", "c", "d"],
                }
            ),
            "mean",
            pd.DataFrame(
                {"input_token_pos": [0, 1], "attr_score": [1.5, 3.5], "other_col": ["b", "d"]}
            ),
        ),
        # Test case 5: Single row dataframe
        (
            pd.DataFrame({"input_token_pos": [0], "attr_score": [1], "other_col": ["a"]}),
            "sum",
            pd.DataFrame({"input_token_pos": [0], "attr_score": [1], "other_col": ["a"]}),
        ),
    ],
)
def test_aggregate_attr_score_df(experiment_logger, input_df, score_agg, expected_df):
    result_df = experiment_logger._aggregate_attr_score_df(input_df, score_agg)
    pd.testing.assert_frame_equal(result_df, expected_df)
