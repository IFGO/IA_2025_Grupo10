import pytest
import pandas as pd
import os
import logging

from sklearn.datasets import make_regression
from src.model_training import (
    train_and_evaluate_model,  # type: ignore
    compare_models,  # type: ignore
    get_best_model_by_mse,  # type: ignore
    limpar_modelos_antigos,
)


@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)  # type: ignore
    df_X = pd.DataFrame(X, columns=["feat1", "feat2", "feat3"])
    series_y = pd.Series(y)
    return df_X, series_y


@pytest.fixture
def temp_folder(tmp_path):  # type: ignore
    folder = tmp_path / "models"  # type: ignore
    folder.mkdir()  # type: ignore
    return folder  # type: ignore


def test_train_linear_model(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "Linear", kfolds=3, pair_name="test_linear", models_folder=str(temp_folder))  # type: ignore
    assert any(f.name.startswith("linear_test_linear") for f in temp_folder.iterdir())  # type: ignore


def test_train_polynomial_model_valid(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "Polynomial", kfolds=3, pair_name="test_poly", models_folder=str(temp_folder), poly_degree=3)  # type: ignore
    assert any(f.name.startswith("polynomial_test_poly") for f in temp_folder.iterdir())  # type: ignore


def test_train_polynomial_model_invalid_degree(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "Polynomial", kfolds=3, pair_name="test_invalid", models_folder=str(temp_folder), poly_degree=1)  # type: ignore
    assert len(list(temp_folder.iterdir())) == 0  # type: ignore


def test_train_mlp_model(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "MLP", kfolds=3, pair_name="test_mlp", models_folder=str(temp_folder))  # type: ignore
    assert any(f.name.startswith("mlp_test_mlp") for f in temp_folder.iterdir())  # type: ignore


def test_train_randomforest_model(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "RandomForest", kfolds=3, pair_name="test_rf", models_folder=str(temp_folder))  # type: ignore
    assert any(f.name.startswith("randomforest_test_rf") for f in temp_folder.iterdir())  # type: ignore


def test_invalid_model_type(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "InvalidModel", kfolds=3, pair_name="test_invalid", models_folder=str(temp_folder))  # type: ignore
    assert len(list(temp_folder.iterdir())) == 0  # type: ignore


def test_compare_models(sample_data, temp_folder, caplog):  # type: ignore
    X, y = sample_data  # type: ignore
    with caplog.at_level(logging.INFO):  # type: ignore
        compare_models(X, y, kfolds=3, pair_name="test_compare", plots_folder=str(temp_folder))  # type: ignore
    assert "Comparação de Modelos" in caplog.text  # type: ignore


def test_get_best_model_by_mse(sample_data):  # type: ignore
    X, y = sample_data  # type: ignore
    model, name = get_best_model_by_mse(X, y, kfolds=3)  # type: ignore
    assert model is not None
    assert name in {"MLP", "Linear", "Polynomial", "RandomForest"}


def test_limpar_modelos_antigos(temp_folder):  # type: ignore
    # cria arquivos falsos
    nomes = ["mlp", "linear", "polynomial", "randomforest"]
    for nome in nomes:
        path = os.path.join(temp_folder, f"{nome}_BTC_USDT.pkl")  # type: ignore
        open(path, "w").close()

    limpar_modelos_antigos("BTC_USDT", str(temp_folder))  # type: ignore

    assert len(list(temp_folder.iterdir())) == 0  # type: ignore
