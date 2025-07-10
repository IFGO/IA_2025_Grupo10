import pytest
import pandas as pd
import numpy as np
import logging
import os
import joblib  # type: ignore
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from src.model_training import train_and_evaluate_model, compare_models  # type: ignore

# Configura o logging para evitar poluir a saída do teste
logging.basicConfig(level=logging.CRITICAL)


@pytest.fixture
def sample_model_data():
    """
    Cria dados de exemplo para testes de treinamento de modelo.
    """
    np.random.seed(42)
    num_samples = 100
    X_data = pd.DataFrame(
        {
            "feature1": np.random.rand(num_samples) * 10,
            "feature2": np.random.rand(num_samples) * 5,
            "feature3": np.random.rand(num_samples) * 2,
        }
    )
    # Criar um target com alguma relação linear e ruído
    y_data = (
        2 * X_data["feature1"]
        + 0.5 * X_data["feature2"]
        - X_data["feature3"]
        + np.random.randn(num_samples) * 0.5
    )
    y_data = pd.Series(y_data, name="close")
    return X_data, y_data


@pytest.fixture
def temp_models_folder(tmp_path):  # type: ignore
    """
    Cria uma pasta temporária para salvar modelos durante os testes.
    """
    folder = tmp_path / "test_models"  # type: ignore
    folder.mkdir()  # type: ignore
    return str(folder)  # type: ignore


def test_train_and_evaluate_model_mlp(sample_model_data, temp_models_folder):  # type: ignore
    """
    Testa o treinamento e avaliação do MLPRegressor.
    """
    X, y = sample_model_data  # type: ignore
    pair_name = "TEST_MLP"
    model_type = "MLP"
    kfolds = 3

    train_and_evaluate_model(X, y, model_type, kfolds, pair_name, temp_models_folder)  # type: ignore

    # Verifica se o modelo foi salvo
    model_path = os.path.join(temp_models_folder, f"{model_type.lower()}_{pair_name}.pkl")  # type: ignore
    assert os.path.exists(model_path)

    # Tenta carregar o modelo e fazer uma previsão básica
    loaded_model = joblib.load(model_path)  # type: ignore
    assert isinstance(loaded_model, MLPRegressor)
    assert loaded_model.predict(X.iloc[:1]).shape == (1,)  # type: ignore


def test_train_and_evaluate_model_linear(sample_model_data, temp_models_folder):  # type: ignore
    """
    Testa o treinamento e avaliação da Regressão Linear.
    """
    X, y = sample_model_data  # type: ignore
    pair_name = "TEST_LINEAR"
    model_type = "Linear"
    kfolds = 3

    train_and_evaluate_model(X, y, model_type, kfolds, pair_name, temp_models_folder)  # type: ignore

    # Verifica se o modelo foi salvo
    model_path = os.path.join(temp_models_folder, f"{model_type.lower()}_{pair_name}.pkl")  # type: ignore
    assert os.path.exists(model_path)

    # Tenta carregar o modelo e fazer uma previsão básica
    loaded_model = joblib.load(model_path)  # type: ignore
    assert isinstance(loaded_model, LinearRegression)
    assert loaded_model.predict(X.iloc[:1]).shape == (1,)  # type: ignore


def test_compare_models(sample_model_data, tmp_path):  # type: ignore
    """
    Testa a função de comparação de modelos.
    """
    X, y = sample_model_data  # type: ignore
    pair_name = "TEST_COMPARE"
    kfolds = 3
    plots_folder = tmp_path / "test_plots"  # type: ignore
    plots_folder.mkdir()  # type: ignore

    compare_models(X, y, kfolds, pair_name, str(plots_folder))  # type: ignore

    # Verifica se o gráfico de dispersão foi gerado
    plot_path = os.path.join(str(plots_folder), f"scatter_plot_models_{pair_name}.png")  # type: ignore
    assert os.path.exists(plot_path)

    # Verifica se a função executou sem erros graves (verificações mais aprofundadas seriam para métricas)
    # A saída do log pode ser verificada para resultados de comparação, mas isso é mais complexo em testes unitários.


def test_train_and_evaluate_model_with_custom_split(sample_model_data, temp_models_folder):  # type: ignore
    """
    Testa o treinamento com uma fração de validação personalizada (hold-out de 50%).
    """
    X, y = sample_model_data  # type: ignore
    pair_name = "TEST_MLP_SPLIT"
    model_type = "MLP"
    kfolds = 3
    test_size = 0.5  # 50% dos dados para validação final

    # Chamada com parâmetro adicional
    train_and_evaluate_model(
        X,  # type: ignore
        y,  # type: ignore
        model_type=model_type,
        kfolds=kfolds,
        pair_name=pair_name,
        models_folder=temp_models_folder,  # type: ignore
        test_size=test_size,
    )

    model_path = os.path.join(temp_models_folder, f"{model_type.lower()}_{pair_name}.pkl")  # type: ignore
    assert os.path.exists(model_path)

    loaded_model = joblib.load(model_path)  # type: ignore
    assert isinstance(loaded_model, MLPRegressor)
    assert loaded_model.predict(X.iloc[:1]).shape == (1,)  # type: ignore


def test_compare_models_with_custom_split(sample_model_data, tmp_path):  # type: ignore
    """
    Testa a função de comparação de modelos com validação hold-out de 40%.
    """
    X, y = sample_model_data  # type: ignore
    pair_name = "TEST_COMPARE_SPLIT"
    kfolds = 3
    test_size = 0.4
    plots_folder = tmp_path / "test_plots_split"  # type: ignore
    plots_folder.mkdir()  # type: ignore

    compare_models(X, y, kfolds, pair_name, str(plots_folder), test_size=test_size)  # type: ignore

    plot_path = os.path.join(str(plots_folder), f"scatter_plot_models_{pair_name}.png")  # type: ignore
    assert os.path.exists(plot_path)
