import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from src.prediction_profit import simulate_investment_and_profit

"""
    Configura o ambiente temporário para os testes.

    Cria pastas temporárias para modelos e plots, e gera um arquivo JSON de features
    simulado para um par de moedas específico.
"""    
@pytest.fixture
def setup_test_environment(tmp_path):
    pair_name = "BTC_USDT"
    models_folder = tmp_path / "models"
    plots_folder = tmp_path / "plots"
    models_folder.mkdir()
    plots_folder.mkdir()

    features_path = models_folder / f"features_{pair_name}.json"
    features_path.write_text('["feature1"]')

    return {
        "pair_name": pair_name,
        "models_folder": str(models_folder),
        "profit_plots_folder": str(plots_folder),
        "features_path": str(features_path),
        "expected_plot": plots_folder / f"profit_evolution_{pair_name}.png",
        "tmp_path": tmp_path,
    }

"""
    Testa a simulação completa da função simulate_investment_and_profit com sucesso.

    Simula a existência de arquivos necessários, cria dados de entrada fictícios,
    e verifica se o gráfico de evolução do lucro é gerado corretamente.
"""
@patch("src.prediction_profit.joblib.load")
@patch("src.prediction_profit.os.path.exists")
@patch("src.prediction_profit.pd.read_csv")
def test_simulation_runs_and_creates_plot(mock_read_csv, mock_exists, mock_joblib, setup_test_environment):
    def exists_side_effect(path):
        return True  # Simula que tudo existe
    mock_exists.side_effect = exists_side_effect

    # Dados de entrada simulados
    mock_read_csv.return_value = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5),
        "close": [100, 102, 101, 105, 107],
        "feature1": [1, 2, 3, 4, 5],
    })

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([101, 102, 103, 104, 105])
    mock_joblib.return_value = mock_model

    # Criar arquivos de modelo
    for m in ["mlp", "linear", "polynomial", "randomforest"]:
        path = os.path.join(setup_test_environment["models_folder"], f"{m}_{setup_test_environment['pair_name']}.pkl")
        with open(path, "wb") as f:
            f.write(b"fake")

    simulate_investment_and_profit(
        X=None,
        y=None,
        dates=None,
        pair_name=setup_test_environment["pair_name"],
        models_folder=setup_test_environment["models_folder"],
        profit_plots_folder=setup_test_environment["profit_plots_folder"],
    )

    assert setup_test_environment["expected_plot"].exists(), "Gráfico não foi gerado"

"""
    Testa que a simulação aborta corretamente quando não existem arquivos de modelo.

    Simula a existência apenas do arquivo JSON de features, sem modelos de previsão,
    e verifica que nenhum gráfico é gerado.
"""
@patch("src.prediction_profit.os.path.exists")
def test_simulation_aborts_if_no_model(mock_exists, setup_test_environment):
    # Simula que nenhum modelo existe, mas JSON sim
    def exists_side_effect(path):
        return path == setup_test_environment["features_path"]
    mock_exists.side_effect = exists_side_effect

    simulate_investment_and_profit(
        X=None,
        y=None,
        dates=None,
        pair_name=setup_test_environment["pair_name"],
        models_folder=setup_test_environment["models_folder"],
        profit_plots_folder=setup_test_environment["profit_plots_folder"],
    )

    assert not setup_test_environment["expected_plot"].exists(), "Gráfico não deveria ser gerado"

"""
    Testa que a simulação aborta corretamente quando o CSV pré-processado está ausente.

    Simula a existência dos modelos e do arquivo JSON de features, mas ausência do CSV,
    e verifica que nenhum gráfico é gerado.
"""
@patch("src.prediction_profit.os.path.exists")
@patch("src.prediction_profit.joblib.load")
def test_simulation_aborts_if_csv_missing(mock_joblib, mock_exists, setup_test_environment):
    # Cria modelos, mas CSV falta
    def exists_side_effect(path):
        if path.endswith(".pkl"):
            return True
        elif path.endswith(".json"):
            return True
        return False
    mock_exists.side_effect = exists_side_effect

    # Criar arquivos de modelo
    for m in ["mlp", "linear", "polynomial", "randomforest"]:
        path = os.path.join(setup_test_environment["models_folder"], f"{m}_{setup_test_environment['pair_name']}.pkl")
        with open(path, "wb") as f:
            f.write(b"fake")

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([101, 102, 103, 104, 105])
    mock_joblib.return_value = mock_model

    simulate_investment_and_profit(
        X=None,
        y=None,
        dates=None,
        pair_name=setup_test_environment["pair_name"],
        models_folder=setup_test_environment["models_folder"],
        profit_plots_folder=setup_test_environment["profit_plots_folder"],
    )

    assert not setup_test_environment["expected_plot"].exists(), "Gráfico não deveria ser gerado"
