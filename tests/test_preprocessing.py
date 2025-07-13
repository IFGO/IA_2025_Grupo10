import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_features, remove_high_vif_features

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.normal(size=100),
        "feature2": np.random.normal(size=100),
        "feature3": np.random.normal(size=100)
    })
    X["feature4"] = X["feature1"] * 0.9 + np.random.normal(scale=0.1, size=100)  # Colinear
    y = X["feature1"] * 2 + np.random.normal(size=100)
    return X, y

def test_remove_high_vif_features_removes_colinear(sample_data):
    X, _ = sample_data
    X_filtered = remove_high_vif_features(X, threshold=5.0)
    assert "feature4" not in X_filtered.columns or "feature1" not in X_filtered.columns
    assert X_filtered.shape[1] < X.shape[1]

def test_remove_high_vif_features_no_removal(sample_data):
    X, _ = sample_data
    X = X.drop(columns="feature4")  # Removendo colinear antes
    X_filtered = remove_high_vif_features(X, threshold=5.0)
    assert X_filtered.shape[1] == X.shape[1]

def test_preprocess_features_basic(sample_data):
    X, y = sample_data
    result = preprocess_features(X, y, vif_threshold=5.0, k_best=2)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] <= 3  # uma removida por VIF, duas escolhidas

def test_preprocess_features_force_include(sample_data):
    X, y = sample_data
    col_forced = "feature4"
    result = preprocess_features(X, y, vif_threshold=5.0, k_best=2, force_include=[col_forced])
    assert col_forced in result.columns

def test_preprocess_features_kbest_limit(sample_data):
    X, y = sample_data
    result = preprocess_features(X, y, vif_threshold=5.0, k_best=20)  # maior que total de colunas
    assert result.shape[1] <= X.shape[1]
