import pytest
import pandas as pd
import numpy as np
import os
import logging
from src.data_analyzer import calculate_statistics, generate_analysis_plots, calculate_comparative_variability  # type: ignore

# Configura o logging para evitar poluir a saída do teste
logging.basicConfig(level=logging.CRITICAL)

"""
    Cria um DataFrame de exemplo com dados simulados financeiros (datas, preços e volumes)
    para usar nos testes das funções do módulo data_analyzer.
"""
@pytest.fixture
def sample_analyzer_df():
    """
    Cria um DataFrame de exemplo para testes de data_analyzer.
    """
    data = {  # type: ignore
        "date": pd.to_datetime(pd.date_range(start="2023-01-01", periods=50, freq="D")),  # type: ignore
        "close": np.random.rand(50) * 100 + 50,
        "open": np.random.rand(50) * 100 + 45,
        "high": np.random.rand(50) * 100 + 55,
        "low": np.random.rand(50) * 100 + 40,
        "volume": np.random.rand(50) * 1000000 + 10000,
    }
    return pd.DataFrame(data)

"""
    Cria um dicionário contendo dois DataFrames simulados, representando dados de
    duas criptomoedas diferentes, para testes comparativos.
"""
@pytest.fixture
def sample_all_data_dict(sample_analyzer_df):  # type: ignore

    df1 = sample_analyzer_df.copy()  # type: ignore
    df2 = sample_analyzer_df.copy()  # type: ignore
    df2["close"] = df2["close"] * 1.2  # Diferenciar um pouco
    return {"BTC_USDT": df1, "ETH_USDT": df2}  # type: ignore

"""
    Testa se a função calculate_statistics calcula corretamente as estatísticas descritivas
    (média, desvio padrão, variância, assimetria, curtose) sobre o DataFrame de exemplo.
    Verifica também se o número de elementos contado está correto.
"""
def test_calculate_statistics(sample_analyzer_df):  # type: ignore

    stats = calculate_statistics(sample_analyzer_df)  # type: ignore

    assert "mean" in stats
    assert "std" in stats
    assert "variance" in stats
    assert "skewness" in stats
    assert "kurtosis" in stats
    assert stats["count"] == len(sample_analyzer_df)  # type: ignore
    assert np.isclose(stats["mean"], sample_analyzer_df["close"].mean())  # type: ignore
    assert np.isclose(stats["std"], sample_analyzer_df["close"].std())  # type: ignore

"""
    Testa se a função generate_analysis_plots gera um gráfico de análise a partir dos dados
    de exemplo, salva o arquivo PNG no diretório temporário e se esse arquivo existe e não está vazio.
"""
def test_generate_analysis_plots(sample_analyzer_df, tmp_path):  # type: ignore
    
    save_folder = tmp_path / "analysis_plots"  # type: ignore
    save_folder.mkdir()  # type: ignore
    pair_name = "TEST_PAIR"

    generate_analysis_plots(sample_analyzer_df, pair_name, str(save_folder))  # type: ignore

    plot_path = os.path.join(str(save_folder), f"analise_{pair_name}.png")  # type: ignore
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0  # Verifica se o arquivo não está vazio

"""
    Testa se a função calculate_comparative_variability calcula corretamente a variabilidade relativa
    (coeficiente de variação) para cada criptomoeda no dicionário de dados, e se o DataFrame resultante
    possui as colunas esperadas e os dados coerentes.
"""
def test_calculate_comparative_variability(sample_all_data_dict):  # type: ignore
    
    df_variability = calculate_comparative_variability(sample_all_data_dict)  # type: ignore

    assert not df_variability.empty
    assert "Criptomoeda" in df_variability.columns
    assert "Coef. de Variação (%) (Variabilidade Relativa)" in df_variability.columns
    assert len(df_variability) == len(sample_all_data_dict)  # type: ignore
    assert df_variability.iloc[0]["Criptomoeda"] in [
        "BTC USDT",
        "ETH USDT",
    ]  # Ordem pode variar
    assert df_variability.iloc[1]["Criptomoeda"] in ["BTC USDT", "ETH USDT"]
    assert df_variability.iloc[0]["Coef. de Variação (%) (Variabilidade Relativa)"] >= 0
