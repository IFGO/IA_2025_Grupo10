# tests/test_data_analyzer.py
import pytest
import pandas as pd
import numpy as np
import os
import logging
from src.data_analyzer import calculate_statistics, generate_analysis_plots, calculate_comparative_variability

# Configura o logging para evitar poluir a saída do teste
logging.basicConfig(level=logging.CRITICAL)

@pytest.fixture
def sample_analyzer_df():
    """
    Cria um DataFrame de exemplo para testes de data_analyzer.
    """
    data = {
        'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='D')),
        'close': np.random.rand(50) * 100 + 50,
        'open': np.random.rand(50) * 100 + 45,
        'high': np.random.rand(50) * 100 + 55,
        'low': np.random.rand(50) * 100 + 40,
        'volume': np.random.rand(50) * 1000000 + 10000
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_all_data_dict(sample_analyzer_df):
    """
    Cria um dicionário de DataFrames para testes comparativos.
    """
    df1 = sample_analyzer_df.copy()
    df2 = sample_analyzer_df.copy()
    df2['close'] = df2['close'] * 1.2 # Diferenciar um pouco
    return {"BTC_USDT": df1, "ETH_USDT": df2}

def test_calculate_statistics(sample_analyzer_df):
    """
    Testa se o cálculo das estatísticas funciona e retorna as esperadas.
    """
    stats = calculate_statistics(sample_analyzer_df)

    assert 'mean' in stats
    assert 'std' in stats
    assert 'variance' in stats
    assert 'skewness' in stats
    assert 'kurtosis' in stats
    assert stats['count'] == len(sample_analyzer_df)
    assert np.isclose(stats['mean'], sample_analyzer_df['close'].mean())
    assert np.isclose(stats['std'], sample_analyzer_df['close'].std())

def test_generate_analysis_plots(sample_analyzer_df, tmp_path):
    """
    Testa se a geração de gráficos de análise funciona e salva o arquivo.
    """
    save_folder = tmp_path / "analysis_plots"
    save_folder.mkdir()
    pair_name = "TEST_PAIR"

    generate_analysis_plots(sample_analyzer_df, pair_name, str(save_folder))

    plot_path = os.path.join(str(save_folder), f"analise_{pair_name}.png")
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0 # Verifica se o arquivo não está vazio

def test_calculate_comparative_variability(sample_all_data_dict):
    """
    Testa o cálculo da variabilidade comparativa.
    """
    df_variability = calculate_comparative_variability(sample_all_data_dict)

    assert not df_variability.empty
    assert 'Criptomoeda' in df_variability.columns
    assert 'Coef. de Variação (%) (Variabilidade Relativa)' in df_variability.columns
    assert len(df_variability) == len(sample_all_data_dict)
    assert df_variability.iloc[0]['Criptomoeda'] in ["BTC USDT", "ETH USDT"] # Ordem pode variar
    assert df_variability.iloc[1]['Criptomoeda'] in ["BTC USDT", "ETH USDT"]
    assert df_variability.iloc[0]['Coef. de Variação (%) (Variabilidade Relativa)'] >= 0
