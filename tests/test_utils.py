import os
import pytest
import shutil
import builtins
from unittest import mock
from src import utils
from pathlib import Path
import sys
import types

@pytest.fixture
def mock_config():
    with mock.patch("src.utils.MOEDA_COTACAO", "USDT"), \
         mock.patch("src.utils.TIMEFRAME", "1h"), \
         mock.patch("src.utils.OUTPUT_FOLDER", "data/output"), \
         mock.patch("src.utils.PROCESSED_DATA_FOLDER", "data/processed"), \
         mock.patch("src.utils.MODELS_FOLDER", "models"), \
         mock.patch("src.utils.RAW_FILENAME_TEMPLATE", "{base}_{quote}_{timeframe}.csv"), \
         mock.patch("src.utils.FEATURED_FILENAME_TEMPLATE", "{base}_{quote}_features.csv"), \
         mock.patch("src.utils.MODEL_FILENAME_TEMPLATE", "{model_type}_{base}_{quote}.pkl"):
        yield


def test_get_pair_key(mock_config):
    result = utils.get_pair_key("btc")
    assert result == "BTC_USDT"


def test_get_raw_data_filepath(mock_config):
    expected = os.path.join("data/output", "BTC_USDT_1h.csv")
    assert utils.get_raw_data_filepath("btc") == expected


def test_get_processed_data_filepath(mock_config):
    expected = os.path.join("data/processed", "BTC_USDT_features.csv")
    assert utils.get_processed_data_filepath("btc") == expected


def test_get_model_filepath(mock_config):
    expected = os.path.join("models", "mlp_BTC_USDT.pkl")
    assert utils.get_model_filepath("mlp", "btc") == expected

# criar mockup para variaveis globais

@pytest.fixture
def mock_config_for_limpar(tmp_path):
    # Criar módulo fake config com pastas dentro do tmp_path
    fake_config = types.SimpleNamespace(
        OUTPUT_FOLDER=str(tmp_path / "output"),
        PROCESSED_DATA_FOLDER=str(tmp_path / "processed"),
        MODELS_FOLDER=str(tmp_path / "models"),
        PLOTS_FOLDER=str(tmp_path / "plots"),
        ANALYSIS_FOLDER=str(tmp_path / "analysis"),
        PROFIT_PLOTS_FOLDER=str(tmp_path / "profit_plots"),
        STATS_REPORTS_FOLDER=str(tmp_path / "stats_reports"),
    )
    sys.modules["config"] = fake_config
    # Criar as pastas e arquivos dummy
    for folder in vars(fake_config).values():
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        (p / "dummy.txt").write_text("teste")
    yield fake_config
    # Opcional: remover módulo fake depois
    sys.modules.pop("config", None)


def test_limpar_pastas_saida(mock_config_for_limpar):
    utils.limpar_pastas_saida()
    # Verificar que as pastas estão vazias
    for pasta in vars(mock_config_for_limpar).values():
        p = Path(pasta)
        assert all(f.is_file() is False for f in p.iterdir()) or not any(p.iterdir())