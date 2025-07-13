import os
import pytest
import shutil
import builtins
from unittest import mock
from src import utils
from pathlib import Path
import sys
import types

"""
    Aplica mocks nas variáveis globais de configuração do módulo `utils`,
    como nomes de moeda, pastas e templates de nomes de arquivos.

    Isso permite testes independentes do ambiente real de configuração.
"""
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

"""
    Testa a função `get_pair_key` para garantir que o par de moedas seja retornado
    corretamente no formato esperado (exemplo: "btc" → "BTC_USDT").
"""
def test_get_pair_key(mock_config):
    result = utils.get_pair_key("btc")
    assert result == "BTC_USDT"

"""
    Testa a função `get_raw_data_filepath` para validar que o caminho do arquivo bruto
    (raw data) é construído corretamente com base no par de moedas e configurações.
"""
def test_get_raw_data_filepath(mock_config):
    expected = os.path.join("data/output", "BTC_USDT_1h.csv")
    assert utils.get_raw_data_filepath("btc") == expected

"""
    Testa a função `get_processed_data_filepath` para verificar se o caminho do arquivo
    processado (features) é gerado corretamente conforme o padrão configurado.
"""
def test_get_processed_data_filepath(mock_config):
    expected = os.path.join("data/processed", "BTC_USDT_features.csv")
    assert utils.get_processed_data_filepath("btc") == expected

"""
    Testa a função `get_model_filepath` para garantir que o caminho do arquivo de modelo
    (.pkl) é criado corretamente de acordo com o tipo do modelo e o par de moedas.
"""
def test_get_model_filepath(mock_config):
    expected = os.path.join("models", "mlp_BTC_USDT.pkl")
    assert utils.get_model_filepath("mlp", "btc") == expected

# criar mockup para variaveis globais
"""
    Cria um módulo de configuração fake com múltiplas pastas temporárias para simular
    o ambiente de saída usado pelo utilitário `limpar_pastas_saida`.

    Também cria arquivos dummy em cada pasta para testar a limpeza.
"""
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

"""
    Testa a função `limpar_pastas_saida` para verificar se ela limpa corretamente
    todos os arquivos dentro das pastas definidas na configuração fake.
"""    
def test_limpar_pastas_saida(mock_config_for_limpar):
    utils.limpar_pastas_saida()
    # Verificar que as pastas estão vazias
    for pasta in vars(mock_config_for_limpar).values():
        p = Path(pasta)
        assert all(f.is_file() is False for f in p.iterdir()) or not any(p.iterdir())