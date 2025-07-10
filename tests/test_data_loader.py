import pandas as pd
import logging
from pathlib import Path
from src.data_loader import load_crypto_data

# Configura o logging para evitar poluir a saída do teste
logging.basicConfig(level=logging.CRITICAL)

# Removido mock_csv_data fixture, pois o mock_read_csv agora retorna o DataFrame diretamente


def test_load_crypto_data_success(monkeypatch):  # type: ignore
    """
    Testa o carregamento bem-sucedido de dados de criptomoeda.
    """

    def mock_read_csv(filepath, skiprows=1, encoding=None):  # type: ignore
        if Path(filepath).name == "BTC_USDT_d.csv":  # type: ignore
            data = {  # type: ignore
                "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),  # type: ignore
                "close": [100.0, 101.5, 102.0],
                "open": [99.0, 100.0, 101.5],
            }
            return pd.DataFrame(data)
        raise Exception(f"Arquivo mock inesperado: {filepath}")

    # Patching correto
    monkeypatch.setattr("src.data_loader.pd.read_csv", mock_read_csv)  # type: ignore
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)  # type: ignore

    df = load_crypto_data(base_symbol="BTC", quote_symbol="USDT", timeframe="d")

    assert df is not None
    assert not df.empty
    assert "date" in df.columns
    assert "close" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])  # type: ignore
    assert pd.api.types.is_numeric_dtype(df["close"])  # type: ignore
    assert len(df) == 3
    assert df["close"].iloc[0] == 100.0  # type: ignore


def test_load_crypto_data_empty_data(monkeypatch):  # type: ignore
    """
    Testa o carregamento de dados vazios.
    """

    def mock_read_csv_empty(url, skiprows=1):  # type: ignore
        return pd.DataFrame(columns=["date", "close"])  # Retorna DataFrame vazio

    monkeypatch.setattr(pd, "read_csv", mock_read_csv_empty)  # type: ignore

    df = load_crypto_data(base_symbol="BTC", quote_symbol="USDT", timeframe="d")
    assert df is None  # Espera None para DataFrame vazio


def test_load_crypto_data_http_error(monkeypatch):  # type: ignore
    """
    Testa o tratamento de erro HTTP (ex: 404 Not Found).
    """

    def mock_read_csv_http_error(url, skiprows=1):  # type: ignore
        from urllib.error import HTTPError

        raise HTTPError(url, 404, "Not Found", {}, None)  # type: ignore

    monkeypatch.setattr(pd, "read_csv", mock_read_csv_http_error)  # type: ignore

    df = load_crypto_data(base_symbol="BTC", quote_symbol="USDT", timeframe="d")
    assert df is None


def test_load_crypto_data_missing_close_column(monkeypatch):  # type: ignore
    """
    Testa o cenário onde a coluna 'close' está faltando.
    """

    def mock_read_csv_no_close(url, skiprows=1):  # type: ignore
        data = {"date": pd.to_datetime(["2023-01-01"]), "open": [100.0]}  # type: ignore
        return pd.DataFrame(data)

    monkeypatch.setattr(pd, "read_csv", mock_read_csv_no_close)  # type: ignore

    df = load_crypto_data(base_symbol="TEST", quote_symbol="USDT", timeframe="d")
    assert df is None
