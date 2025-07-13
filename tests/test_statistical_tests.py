import pytest
import pandas as pd
import numpy as np
import os
import shutil
from src import statistical_tests as sa

"""
    Cria um diretório temporário chamado 'reports' para armazenar relatórios gerados
    durante os testes estatísticos.
"""
@pytest.fixture
def setup_folder(tmp_path):
    path = tmp_path / "reports"
    path.mkdir()
    return str(path)

"""
    Testa a função `_calculate_daily_returns` com um DataFrame de preços válidos.

    Verifica:
        - Se o número de retornos calculados está correto.
        - Se os valores retornados estão corretos em relação ao cálculo manual esperado.
"""
def test_calculate_daily_returns_valid():
    prices = pd.DataFrame({"close": [100, 102, 101, 103]})
    returns = sa._calculate_daily_returns(prices).reset_index(drop=True)

    # Cálculo manual esperado:
    expected = pd.Series([(102-100)/100, (101-102)/102, (103-101)/101])

    assert len(returns) == len(expected)
    assert all(np.isclose(returns, expected, rtol=1e-5))

"""
    Testa a função `_calculate_daily_returns` com um DataFrame vazio.

    Verifica:
        - Se o retorno da função também é vazio.
"""
def test_calculate_daily_returns_empty():
    df = pd.DataFrame({"close": []})
    returns = sa._calculate_daily_returns(df)
    assert returns.empty

"""
    Testa a função `_calculate_daily_returns` quando a coluna 'close' está ausente.

    Verifica:
        - Se a função retorna um DataFrame vazio em caso de estrutura inválida.
"""
def test_calculate_daily_returns_missing_column():
    df = pd.DataFrame({"open": [1, 2, 3]})
    returns = sa._calculate_daily_returns(df)
    assert returns.empty

"""
    Testa a função `perform_hypothesis_test` com dados simulados válidos.

    Verifica:
        - Se o relatório de teste de hipótese é gerado corretamente.
        - Se o conteúdo do relatório contém o termo "Retorno Médio da Amostra".
"""
def test_perform_hypothesis_test_creates_report(setup_folder):
    np.random.seed(0)
    returns = np.random.normal(0.001, 0.01, 100)
    prices = 100 * (1 + pd.Series(returns)).cumprod()
    df = pd.DataFrame({"close": prices})

    sa.perform_hypothesis_test(
        df=df,
        pair_name="TEST_COIN",
        target_return_percent=0.0005,
        save_folder=setup_folder,
        alpha=0.05,
    )

    report_path = os.path.join(setup_folder, "hypothesis_test_report_TEST_COIN.txt")
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Retorno Médio da Amostra" in content

"""
    Testa a função `perform_hypothesis_test` com um DataFrame vazio.

    Verifica:
        - Se nenhum relatório é gerado quando os dados são insuficientes.
"""
def test_perform_hypothesis_test_empty_df(setup_folder):
    df = pd.DataFrame({"close": []})
    sa.perform_hypothesis_test(
        df=df,
        pair_name="EMPTY",
        target_return_percent=0.001,
        save_folder=setup_folder,
    )
    report_path = os.path.join(setup_folder, "hypothesis_test_report_EMPTY.txt")
    assert not os.path.exists(report_path)  # nada deve ser gerado

"""
    Testa a função `perform_anova_analysis` com dados simulados para 3 criptomoedas
    com diferentes perfis de retorno médio.

    Verifica:
        - Se os relatórios de ANOVA e os gráficos do teste de Tukey são gerados corretamente.
        - Se todos os arquivos esperados estão presentes na pasta de saída.
"""
def test_perform_anova_analysis_creates_reports(setup_folder):
    np.random.seed(42)
    days = 200

    mock_data = {
        "BTC_USDT": pd.DataFrame({"close": 100 * (1 + np.random.normal(0.004, 0.01, days)).cumprod()}),  # Retorno médio mais alto
        "ETH_USDT": pd.DataFrame({"close": 100 * (1 + np.random.normal(0.001, 0.01, days)).cumprod()}),  # Retorno médio médio
        "ADA_USDT": pd.DataFrame({"close": 100 * (1 + np.random.normal(-0.001, 0.01, days)).cumprod()}),  # Retorno médio negativo
    }

    sa.perform_anova_analysis(all_data=mock_data, save_folder=setup_folder, alpha=0.05)

    expected_files = [
        "anova_report_all_cryptos.txt",
        "anova_report_volatility_groups.txt",
        "tukey_hsd_all_cryptos.png",
        "tukey_hsd_volatility_groups.png",
    ]

    missing_files = []
    for filename in expected_files:
        full_path = os.path.join(setup_folder, filename)
        if not os.path.exists(full_path):
            missing_files.append(filename)

    assert not missing_files, f"Os seguintes arquivos esperados não foram gerados: {missing_files}"

"""
    Testa a função `perform_anova_analysis` com dados insuficientes (DataFrames vazios).

    Verifica:
        - Se nenhum relatório é gerado quando os dados de entrada não permitem análise estatística.
"""
def test_perform_anova_analysis_insufficient_data(setup_folder):
    mock_data = {
        "BTC_USDT": pd.DataFrame({"close": []}),
        "ETH_USDT": pd.DataFrame({"close": []}),
    }

    sa.perform_anova_analysis(all_data=mock_data, save_folder=setup_folder)
    assert not os.path.exists(os.path.join(setup_folder, "anova_report_all_cryptos.txt"))
