import pytest
import pandas as pd
import numpy as np
import os
import logging
from src.data_visualizer import plot_crypto_data

# Configura o logging para evitar poluir a saída do teste
logging.basicConfig(level=logging.CRITICAL)

"""
    Cria um DataFrame de exemplo com 50 dias de dados simulados de fechamento de preços ('close').
    É utilizado como entrada para testar funções de visualização.
"""
@pytest.fixture
def sample_visualizer_df():

    data = {  # type: ignore
        "date": pd.to_datetime(pd.date_range(start="2023-01-01", periods=50, freq="D")),  # type: ignore
        "close": np.random.rand(50) * 100 + 50,
    }
    return pd.DataFrame(data)

"""
    Testa se a função `plot_crypto_data` gera e salva corretamente um gráfico a partir do DataFrame fornecido.

    - Cria uma pasta temporária para salvar o gráfico.
    - Chama a função com dados válidos.
    - Verifica se o arquivo PNG foi criado e se não está vazio.
"""
def test_plot_crypto_data(sample_visualizer_df, tmp_path):  # type: ignore

    save_folder = tmp_path / "simple_plots"  # type: ignore
    save_folder.mkdir()  # type: ignore
    pair_name = "TEST_PAIR_SIMPLE"

    # A função plot_crypto_data agora espera o DataFrame diretamente
    plot_crypto_data(sample_visualizer_df, pair_name, str(save_folder))  # type: ignore

    plot_path = os.path.join(str(save_folder), f"{pair_name.replace(' ', '_')}_chart.png")  # type: ignore # Adicionado replace para consistência
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0  # Verifica se o arquivo não está vazio

"""
    Testa o comportamento da função ao receber um DataFrame vazio.

    - Cria um DataFrame sem dados (linhas vazias).
    - Usa o `caplog` para capturar mensagens de log.
    - Verifica se a função gera um aviso de que o DataFrame está vazio.
    - Garante que nenhum arquivo de imagem seja criado.
"""
def test_plot_crypto_data_empty_df(tmp_path, caplog):  # type: ignore # Adicionado caplog fixture

    save_folder = tmp_path / "simple_plots_empty"  # type: ignore
    save_folder.mkdir()  # type: ignore
    pair_name = "EMPTY_PAIR"
    empty_df = pd.DataFrame({"date": [], "close": []})

    # Captura logs para verificar o warning
    with caplog.at_level(logging.WARNING):  # type: ignore # Usando caplog.at_level
        plot_crypto_data(empty_df, pair_name, str(save_folder))  # type: ignore

    assert "DataFrame vazio ou sem dados válidos para plotar" in caplog.text  # type: ignore
    plot_path = os.path.join(str(save_folder), f"{pair_name.replace(' ', '_')}_chart.png")  # type: ignore # Adicionado replace
    assert not os.path.exists(plot_path)  # O arquivo não deve ser criado
