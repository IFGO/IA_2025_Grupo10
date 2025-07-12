# test_prediction_profit.py
import pandas as pd
import numpy as np
import os
import joblib
import pytest
from unittest.mock import patch, MagicMock
from src.prediction_profit import simulate_investment_and_profit

class MockModel:
    def predict(self, X):
        # Esta implementação precisa ser mais robusta para lidar com diferentes tamanhos de X
        # Para o teste 'success', podemos simular uma correlação com os preços de fechamento.
        # No entanto, para ser genérico para qualquer X, é melhor retornar algo baseado no tamanho de X.
        # Se X estiver vazio, retorna um array vazio.
        if X.empty:
            return np.array([])
        # Caso contrário, retorne previsões do mesmo tamanho de X.
        # Aqui, vamos criar previsões simples baseadas no índice para garantir o tamanho.
        # Em um cenário real, você teria uma lógica de predição mais sofisticada.
        return np.random.rand(len(X)) * 100 + 50 # Retorna valores aleatórios para simular predições


# Define MockModelForEmptyData no escopo global
class MockModelForEmptyData:
    def predict(self, X):
        return np.array([]) # Retorna um array vazio para entrada vazia


@pytest.fixture
def setup_data_and_folders(tmp_path):
    # Cria dados ficticios
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10))
    close_prices = pd.Series(np.random.rand(10) * 100 + 50, name="close")
    # Features precisa ser dataframe
    features = pd.DataFrame(np.random.rand(10, 5), columns=[f"feature_{i}" for i in range(5)])

    # Cria pastas ficticias para models e profit_plots
    models_folder = tmp_path / "models"
    profit_plots_folder = tmp_path / "profit_plots"
    models_folder.mkdir()
    profit_plots_folder.mkdir()

    # Cria modelos ficticios
    model_types = ["mlp", "linear", "randomforest"]
    for m_type in model_types:
        # Agora usando a MockModel definida globalmente
        joblib.dump(MockModel(), models_folder / f"{m_type}_TEST_PAIR.pkl")

    yield dates, close_prices, features, models_folder, profit_plots_folder

def test_simulate_investment_and_profit_success(setup_data_and_folders):
    dates, close_prices, features, models_folder, profit_plots_folder = setup_data_and_folders
    pair_name = "TEST_PAIR"
    initial_investment = 1000.0

    # Mocka matplotlib.pyplot.savefig para prevenir erros de acesso negado C:
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.show'), \
         patch('matplotlib.pyplot.close'):

        simulate_investment_and_profit(
            X=features,
            y=close_prices,
            dates=dates,
            pair_name=pair_name,
            models_folder=models_folder,
            profit_plots_folder=profit_plots_folder,
            initial_investment=initial_investment,
        )

        # Tentativa de salvar fig
        expected_plot_path = os.path.join(profit_plots_folder, f"profit_evolution_{pair_name}.png")
        mock_savefig.assert_called_once_with(expected_plot_path, dpi=150)

        # Checa se o diretorio de profit_plots foi criado
        assert os.path.exists(profit_plots_folder)
        assert os.path.isdir(profit_plots_folder)

def test_simulate_investment_and_profit_no_models_found(tmp_path, caplog):
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10))
    close_prices = pd.Series(np.random.rand(10) * 100 + 50, name="close")
    features = pd.DataFrame(np.random.rand(10, 5), columns=[f"feature_{i}" for i in range(5)])

    models_folder = tmp_path / "empty_models"
    profit_plots_folder = tmp_path / "profit_plots"
    models_folder.mkdir() # Cria pasta vazia

    with caplog.at_level(os.environ.get("LOG_LEVEL", "INFO")): # Usa os.environ.get para captura de erro
        simulate_investment_and_profit(
            X=features,
            y=close_prices,
            dates=dates,
            pair_name="NO_MODEL_PAIR",
            models_folder=models_folder,
            profit_plots_folder=profit_plots_folder,
            initial_investment=1000.0,
        )
    assert "Nenhum modelo carregado para NO_MODEL_PAIR. Simulação cancelada." in caplog.text

def test_simulate_investment_and_profit_invalid_model_file(tmp_path, caplog):
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10))
    close_prices = pd.Series(np.random.rand(10) * 100 + 50, name="close")
    features = pd.DataFrame(np.random.rand(10, 5), columns=[f"feature_{i}" for i in range(5)])

    models_folder = tmp_path / "invalid_models"
    profit_plots_folder = tmp_path / "profit_plots"
    models_folder.mkdir()

    # Cria modelo vazio
    with open(models_folder / "mlp_INVALID_PAIR.pkl", "w") as f:
        f.write("This is not a valid pickle file")

    with caplog.at_level(os.environ.get("LOG_LEVEL", "INFO")):
        simulate_investment_and_profit(
            X=features,
            y=close_prices,
            dates=dates,
            pair_name="INVALID_PAIR",
            models_folder=models_folder,
            profit_plots_folder=profit_plots_folder,
            initial_investment=1000.0,
        )
    assert "Falha ao carregar o modelo mlp" in caplog.text


def test_simulate_investment_and_profit_empty_data(tmp_path, caplog):
    # Testa com inputs de dataframes vazios
    dates = pd.Series(dtype='datetime64[ns]')
    close_prices = pd.Series(dtype='float64', name="close")
    features = pd.DataFrame() # Empty DataFrame

    models_folder = tmp_path / "models"
    profit_plots_folder = tmp_path / "profit_plots"
    models_folder.mkdir()
    profit_plots_folder.mkdir()

    # Uso global do MockModelForEmptyData
    joblib.dump(MockModelForEmptyData(), models_folder / f"mlp_EMPTY_PAIR.pkl")

    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.show'), \
         patch('matplotlib.pyplot.close'):
        simulate_investment_and_profit(
            X=features,
            y=close_prices,
            dates=dates,
            pair_name="EMPTY_PAIR",
            models_folder=models_folder,
            profit_plots_folder=profit_plots_folder,
            initial_investment=1000.0,
        )
        # Deve salvar o plot mesmo que vazio
        expected_plot_path = os.path.join(profit_plots_folder, f"profit_evolution_EMPTY_PAIR.png")
        mock_savefig.assert_called_once_with(expected_plot_path, dpi=150)