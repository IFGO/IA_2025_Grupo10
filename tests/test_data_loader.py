# tests/test_data_loader.py
import pytest
import pandas as pd
import os
import logging
from src.data_loader import load_crypto_data

# Configura o logging para evitar poluir a saída do teste
logging.basicConfig(level=logging.CRITICAL)

# Removido mock_csv_data fixture, pois o mock_read_csv agora retorna o DataFrame diretamente

def test_load_crypto_data_success(monkeypatch):
    """
    Testa o carregamento bem-sucedido de dados de criptomoeda.
    """
    def mock_read_csv(url, skiprows=1):
        # Retorna diretamente um DataFrame que mimetiza a estrutura esperada
        if "Poloniex_BTCUSDT_d.csv" in url:
            data = {
                'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
                'close': [100.0, 101.5, 102.0],
                'open': [99.0, 100.0, 101.5]
            }
            return pd.DataFrame(data)
        raise Exception(f"URL de mock não esperada: {url}")

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

    df = load_crypto_data(base_symbol="BTC", quote_symbol="USDT", timeframe="d")

    assert df is not None
    assert not df.empty
    assert 'date' in df.columns
    assert 'close' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert pd.api.types.is_numeric_dtype(df['close'])
    assert len(df) == 3
    assert df['close'].iloc[0] == 100.0

def test_load_crypto_data_empty_data(monkeypatch):
    """
    Testa o carregamento de dados vazios.
    """
    def mock_read_csv_empty(url, skiprows=1):
        return pd.DataFrame(columns=['date', 'close']) # Retorna DataFrame vazio

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv_empty)

    df = load_crypto_data(base_symbol="BTC", quote_symbol="USDT", timeframe="d")
    assert df is None # Espera None para DataFrame vazio

def test_load_crypto_data_http_error(monkeypatch):
    """
    Testa o tratamento de erro HTTP (ex: 404 Not Found).
    """
    def mock_read_csv_http_error(url, skiprows=1):
        from urllib.error import HTTPError
        raise HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv_http_error)

    df = load_crypto_data(base_symbol="BTC", quote_symbol="USDT", timeframe="d")
    assert df is None

def test_load_crypto_data_missing_close_column(monkeypatch):
    """
    Testa o cenário onde a coluna 'close' está faltando.
    """
    def mock_read_csv_no_close(url, skiprows=1):
        data = {
            'date': pd.to_datetime(['2023-01-01']),
            'open': [100.0]
        }
        return pd.DataFrame(data)

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv_no_close)

    df = load_crypto_data(base_symbol="TEST", quote_symbol="USDT", timeframe="d")
    assert df is None
