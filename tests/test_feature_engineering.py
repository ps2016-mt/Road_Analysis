import pytest
import pandas as pd
import numpy as np
from scripts.feature_engineering import LogTransform

def test_log_transform():
    # Create a sample dataframe
    data = pd.DataFrame({
        "Speed_limit": [30, 50, 70, 90],
        "Number_of_Casualties": [0, 1, 2, 3],
        "Hour_of_Day": [5, 10, 15, 20]
    })

    expected_output = pd.DataFrame({
        "Speed_limit": np.log1p([30, 50, 70, 90]),
        "Number_of_Casualties": np.log1p([0, 1, 2, 3]),
        "Hour_of_Day": np.log1p([5, 10, 15, 20])
    })

    # Initiate the transformer
    transformer = LogTransform(columns=["Speed_limit", "Number_of_Casualties", "Hour_of_Day"])
    transformed_data = transformer.fit_transform(data)

    # Assert equality
    pd.testing.assert_frame_equal(transformed_data, expected_output)
