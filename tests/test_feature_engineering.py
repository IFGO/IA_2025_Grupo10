# tests/test_feature_engineering.py
import pytest
import pandas as pd
import numpy as np
import logging
from src.feature_engineering import create_moving_average_features, create_technical_features

# Configura o logging para evitar poluir a saída do teste
logging.basicConfig(level=logging.CRITICAL)

@pytest.fixture
def sample_dataframe():
    """
    Cria um DataFrame de exemplo para testes de engenharia de features.
    Aumentado o número de amostras para garantir cálculo de features com janelas maiores.
    """
    np.random.seed(42) # Para reprodutibilidade
    num_samples = 1000 # Increased to 1000 to be very safe
    data = {
        'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_samples, freq='D')),
        'close': np.random.rand(num_samples) * 100 + 50, # Preços aleatórios entre 50 e 150
        'open': np.random.rand(num_samples) * 100 + 45,
        'high': np.random.rand(num_samples) * 100 + 55,
        'low': np.random.rand(num_samples) * 100 + 40,
        'volume': np.random.rand(num_samples) * 1000000 + 10000
    }
    return pd.DataFrame(data)

def test_create_moving_average_features(sample_dataframe):
    """
    Testa a criação de features de média móvel e desvio padrão.
    """
    windows = [7, 14]
    df_featured = create_moving_average_features(sample_dataframe.copy(), windows)

    # Verifica se as novas colunas foram criadas
    assert 'sma_7' in df_featured.columns
    assert 'std_7' in df_featured.columns
    assert 'sma_14' in df_featured.columns
    assert 'std_14' in df_featured.columns

    # Verifica que não há NaNs nas partes calculadas das colunas
    # As primeiras 'window - 1' linhas terão NaN, o que é esperado.
    # Verificamos que o restante não é NaN.
    assert not df_featured['sma_7'].iloc[6:].isnull().any() # SMA_7 começa no índice 6 (7º dia)
    assert not df_featured['std_7'].iloc[6:].isnull().any()
    assert not df_featured['sma_14'].iloc[13:].isnull().any() # SMA_14 começa no índice 13 (14º dia)
    assert not df_featured['std_14'].iloc[13:].isnull().any()

    # Verifica um valor de SMA (exemplo manual para sma_7 no 7º dia)
    # O SMA do 7º dia (índice 6) deve ser a média dos primeiros 7 dias.
    expected_sma_7_val = sample_dataframe['close'].iloc[0:7].mean()
    assert np.isclose(df_featured['sma_7'].iloc[6], expected_sma_7_val) # Index 6 é o 7º dia

def test_create_technical_features(sample_dataframe):
    """
    Testa a criação de features técnicas adicionais.
    """
    df_featured = create_technical_features(sample_dataframe.copy())

    # Verifica se as novas colunas foram criadas
    assert 'daily_return' in df_featured.columns
    assert 'volatility_7d' in df_featured.columns
    assert 'volatility_30d' in df_featured.columns
    assert 'close_lag1' in df_featured.columns
    assert 'close_lag5' in df_featured.columns
    assert 'rsi' in df_featured.columns
    assert 'macd' in df_featured.columns
    assert 'macd_signal' in df_featured.columns
    assert 'macd_diff' in df_featured.columns
    assert 'bb_upper' in df_featured.columns
    assert 'bb_lower' in df_featured.columns
    assert 'bb_mavg' in df_featured.columns
    assert 'obv' in df_featured.columns
    assert 'sma_7' in df_featured.columns # Check for SMA/STD from create_moving_average_features
    assert 'std_7' in df_featured.columns
    assert 'sma_14' in df_featured.columns
    assert 'std_14' in df_featured.columns
    assert 'sma_30' in df_featured.columns
    assert 'std_30' in df_featured.columns


    # Verifica se NÃO há NaNs no DataFrame final (após o dropna da função)
    # Esta é a asserção crucial que estava falhando.
    # Usamos .any().any() para verificar se existe QUALQUER NaN em QUALQUER coluna.
    # Adicionando prints para depuração se a asserção falhar
    if df_featured.isnull().any().any():
        print("\n--- DEBUG: NaNs found in df_featured ---")
        print("df_featured.isnull().sum():")
        print(df_featured.isnull().sum()[df_featured.isnull().sum() > 0])
        print("\ndf_featured.head():")
        print(df_featured.head())
        print("\ndf_featured.tail():")
        print(df_featured.tail())
        print("------------------------------------------")

    assert not df_featured.isnull().any().any(), "DataFrame ainda contém NaNs após dropna em create_technical_features."

    # Calcula o comprimento esperado do DataFrame após a remoção de NaNs
    # A maior janela que introduz NaNs é 30 (volatility_30d, sma_30, std_30).
    # A observação empírica anterior indicou que 33 linhas são removidas.
    # Isso significa que o primeiro índice válido é 33.
    first_valid_idx_original_df = 33 # Corresponde ao iloc[0] do df_featured após dropna
    expected_len = len(sample_dataframe) - first_valid_idx_original_df
    assert len(df_featured) == expected_len, f"Comprimento do DataFrame inesperado. Esperado: {expected_len}, Obtido: {len(df_featured)}"

    # Verifica o valor de close_lag1
    # O close_lag1 no índice 0 do df_featured deve ser o close do dia anterior ao primeiro dia válido.
    # Ou seja, sample_dataframe['close'].iloc[first_valid_idx_original_df - 1]
    assert np.isclose(df_featured['close_lag1'].iloc[0], sample_dataframe['close'].iloc[first_valid_idx_original_df - 1])

    # Verifica se daily_return está correto para o primeiro valor não-NaN
    # O primeiro daily_return no df_featured (após dropna) corresponde ao daily_return do primeiro dia válido.
    # Ou seja, (close[first_valid_idx_original_df] - close[first_valid_idx_original_df - 1]) / close[first_valid_idx_original_df - 1]
    expected_daily_return_first = (sample_dataframe['close'].iloc[first_valid_idx_original_df] - sample_dataframe['close'].iloc[first_valid_idx_original_df - 1]) / sample_dataframe['close'].iloc[first_valid_idx_original_df - 1]
    assert np.isclose(df_featured['daily_return'].iloc[0], expected_daily_return_first)

