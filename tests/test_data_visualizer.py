import pytest
import pandas as pd
import numpy as np
import os
import logging
from src.data_visualizer import plot_crypto_data

# Configura o logging para evitar poluir a saída do teste
logging.basicConfig(level=logging.CRITICAL)

@pytest.fixture
def sample_visualizer_df():
    """
    Cria um DataFrame de exemplo para testes de data_visualizer.
    """
    data = {
        'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='D')),
        'close': np.random.rand(50) * 100 + 50
    }
    return pd.DataFrame(data)

def test_plot_crypto_data(sample_visualizer_df, tmp_path):
    """
    Testa se a função de plotagem simples funciona e salva o arquivo.
    """
    save_folder = tmp_path / "simple_plots"
    save_folder.mkdir()
    pair_name = "TEST_PAIR_SIMPLE"

    # A função plot_crypto_data agora espera o DataFrame diretamente
    plot_crypto_data(sample_visualizer_df, pair_name, str(save_folder))

    plot_path = os.path.join(str(save_folder), f"{pair_name.replace(' ', '_')}_chart.png") # Adicionado replace para consistência
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0 # Verifica se o arquivo não está vazio

def test_plot_crypto_data_empty_df(tmp_path, caplog): # Adicionado caplog fixture
    """
    Testa o comportamento da função com um DataFrame vazio.
    Deve emitir um warning e não criar o arquivo.
    """
    save_folder = tmp_path / "simple_plots_empty"
    save_folder.mkdir()
    pair_name = "EMPTY_PAIR"
    empty_df = pd.DataFrame({'date': [], 'close': []})

    # Captura logs para verificar o warning
    with caplog.at_level(logging.WARNING): # Usando caplog.at_level
        plot_crypto_data(empty_df, pair_name, str(save_folder))

    assert "DataFrame vazio ou sem dados válidos para plotar" in caplog.text
    plot_path = os.path.join(str(save_folder), f"{pair_name.replace(' ', '_')}_chart.png") # Adicionado replace
    assert not os.path.exists(plot_path) # O arquivo não deve ser criado
