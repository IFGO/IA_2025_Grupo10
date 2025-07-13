import pandas as pd
import logging
from pathlib import Path
from src.data_loader import load_crypto_data

# Configura o logging para evitar poluir a saída do teste
logging.basicConfig(level=logging.CRITICAL)

# Removido mock_csv_data fixture, pois o mock_read_csv agora retorna o DataFrame diretamente

"""
    Testa o carregamento bem-sucedido de um arquivo CSV com dados de criptomoeda.

    - Cria um arquivo temporário com conteúdo simulado (datas e preços).
    - Usa monkeypatch para simular o caminho do arquivo dentro da função.
    - Verifica se o DataFrame retornado não está vazio, tem colunas esperadas
      e tipos corretos.
"""
def test_load_crypto_data_success(tmp_path,monkeypatch):  # type: ignore

    # mockup alterado para cobrir nao só o read_csv mas também o open()
    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    filepath = data_dir / "BTC_USDT_d.csv"
    content = (
        "date,open,close\n"
        "2023-01-01,99.0,100.0\n"
        "2023-01-02,100.0,101.5\n"
        "2023-01-03,101.5,102.0\n"
    )
    filepath.write_text(content, encoding="utf-8-sig")

    # Monkeypatch Path to return our tmp_path / 'raw' folder whenever Path('data/raw') is called
    original_path_class = Path

    def mock_path(arg=None):
        # Se o código pedir 'data/raw', redirecione para tmp_path / 'raw'
        if arg == "data/raw":
            return data_dir
        # Se pedir o arquivo completo, redirecione para o nosso tmp_path
        if arg == "data/raw/BTC_USDT_d.csv":
            return filepath
        return original_path_class(arg)

    monkeypatch.setattr("src.data_loader.Path", mock_path)

    df = load_crypto_data(base_symbol="BTC", quote_symbol="USDT", timeframe="d")

    assert df is not None
    assert not df.empty
    assert "date" in df.columns
    assert "close" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["close"])
    assert len(df) == 3
    assert df["close"].iloc[0] == 100.0

"""
    Testa o comportamento da função ao carregar um arquivo CSV vazio.

    - Substitui a função `pd.read_csv` por uma versão que retorna um DataFrame vazio.
    - Espera que a função `load_crypto_data` retorne `None` nesse caso.
"""
def test_load_crypto_data_empty_data(monkeypatch):  # type: ignore


    def mock_read_csv_empty(url, skiprows=1):  # type: ignore
        return pd.DataFrame(columns=["date", "close"])  # Retorna DataFrame vazio

    monkeypatch.setattr(pd, "read_csv", mock_read_csv_empty)  # type: ignore

    df = load_crypto_data(base_symbol="BTC", quote_symbol="USDT", timeframe="d")
    assert df is None  # Espera None para DataFrame vazio

"""
    Testa se a função trata corretamente um erro HTTP (como um 404).

    - Simula um erro HTTP ao tentar carregar o CSV.
    - Espera que a função retorne `None` em caso de falha na requisição.
"""
def test_load_crypto_data_http_error(monkeypatch):  # type: ignore


    def mock_read_csv_http_error(url, skiprows=1):  # type: ignore
        from urllib.error import HTTPError

        raise HTTPError(url, 404, "Not Found", {}, None)  # type: ignore

    monkeypatch.setattr(pd, "read_csv", mock_read_csv_http_error)  # type: ignore

    df = load_crypto_data(base_symbol="BTC", quote_symbol="USDT", timeframe="d")
    assert df is None

"""
    Testa o comportamento quando a coluna 'close' está ausente no DataFrame carregado.

    - Simula um arquivo CSV com colunas incompletas.
    - Espera que a função retorne `None` devido à ausência da coluna essencial 'close'.
"""
def test_load_crypto_data_missing_close_column(monkeypatch):  # type: ignore


    def mock_read_csv_no_close(url, skiprows=1):  # type: ignore
        data = {"date": pd.to_datetime(["2023-01-01"]), "open": [100.0]}  # type: ignore
        return pd.DataFrame(data)

    monkeypatch.setattr(pd, "read_csv", mock_read_csv_no_close)  # type: ignore

    df = load_crypto_data(base_symbol="TEST", quote_symbol="USDT", timeframe="d")
    assert df is None
