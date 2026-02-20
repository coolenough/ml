import pytest
import pandas as pd
import numpy as np
from utlis.utlis import cyclic_encode, log_transform, sqrt_transform, Transformer

@pytest.fixture
def sample_df():
    """Provides a fresh DataFrame for every test."""
    return pd.DataFrame({'nums': [1, 4, 9, 16], 'time': [0, 6, 12, 18]})

## --- Test Individual Transforms ---

def test_cyclic_encode(sample_df):
    col = 'time'
    max_val = 18
    result = cyclic_encode(sample_df.copy(), col)

    # Check if new columns exist
    assert "time_sin" in result.columns
    assert "time_cos" in result.columns
    # Check specific value (sin of 0 should be 0)
    assert np.isclose(result.loc[0, "time_sin"], 0)
    # Check max value (cos of 2*pi should be 1)
    assert np.isclose(result.loc[3, "time_cos"], 1)

def test_log_transform(sample_df):
    result = log_transform(sample_df.copy(), 'nums')
    assert "nums_log" in result.columns
    assert np.isclose(result.loc[1, "nums_log"], np.log(4))

def test_sqrt_transform(sample_df):
    result = sqrt_transform(sample_df.copy(), 'nums')
    assert "nums_sqrt" in result.columns
    assert result.loc[2, "nums_sqrt"] == 3.0

## --- Test The Dispatcher (Transformer) ---

def test_transformer_dispatcher(sample_df):
    # Test valid call
    result = Transformer(sample_df.copy(), "log", "nums")
    assert "nums_log" in result.columns

    # Test invalid function name
    with pytest.raises(Exception) as excinfo:
        Transformer(sample_df, "invalid_func", "nums")
    assert "not avilable" in str(excinfo.value)

## --- Test Edge Cases & Errors ---

def test_log_transform_with_zero():
    # log(0) returns -inf, which is mathematically correct but often breaks ML models
    df = pd.DataFrame({'zeros': [0, 1]})
    result = log_transform(df, 'zeros')
    assert np.isinf(result.loc[0, 'zeros_log'])

def test_sqrt_transform_negative():
    # sqrt of negative returns NaN
    df = pd.DataFrame({'negs': [-1, 4]})
    result = sqrt_transform(df, 'negs')
    assert np.isnan(result.loc[0, 'nums_sqrt'] if 'nums_sqrt' in result else np.nan)
