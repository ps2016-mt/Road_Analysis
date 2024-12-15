import pytest
import pandas as pd
import numpy as np
from scripts.feature_engineering import (CustomStandardScaler,
                                         CustomOneHotEncoder)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            pd.DataFrame({"col1": [1, 2, 3],
                          "col2": [4, 5, 6]}),
            pd.DataFrame({"col1": [-1.22474487, 0.0, 1.22474487],
                          "col2": [-1.22474487, 0.0, 1.22474487]}),
        ),
        (
            pd.DataFrame({"col1": [10, 20, 30],
                          "col2": [100, 200, 300]}),
            pd.DataFrame({"col1": [-1.22474487, 0.0, 1.22474487],
                          "col2": [-1.22474487, 0.0, 1.22474487]}),
        ),
    ],
)
def test_custom_standard_scaler(data: pd.DataFrame, expected: pd.DataFrame):
    scaler = CustomStandardScaler()
    transformed_data = scaler.fit_transform(data)
    np.testing.assert_almost_equal(transformed_data,
                                   expected.values,
                                   decimal=6)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            pd.DataFrame({"col1": ["A", "B", "A"],
                          "col2": ["X", "Y", "X"]}),
            pd.DataFrame(
                {
                    "col1_A": [1, 0, 1],
                    "col1_B": [0, 1, 0],
                    "col2_X": [1, 0, 1],
                    "col2_Y": [0, 1, 0],
                }
            ),
        ),
        (
            pd.DataFrame({"col1": ["C", "C", "D"],
                          "col2": ["Z", "Y", "Z"]}),
            pd.DataFrame(
                {
                    "col1_C": [1, 1, 0],
                    "col1_D": [0, 0, 1],
                    "col2_Z": [1, 0, 1],
                    "col2_Y": [0, 1, 0],
                }
            ),
        ),
    ],
)
def test_custom_one_hot_encoder(data, expected):
    encoder = CustomOneHotEncoder()
    transformed_data = encoder.fit_transform(data)
    pd.testing.assert_frame_equal(transformed_data, expected)
