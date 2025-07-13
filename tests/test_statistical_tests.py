import pytest
import pandas as pd
import numpy as np
import os
import shutil
from src import statistical_tests as sa


@pytest.fixture
def setup_folder(tmp_path):
    path = tmp_path / "reports"
    path.mkdir()
    return str(path)


def test_calculate_daily_returns_valid():
    prices = pd.DataFrame({"close": [100, 102, 101, 103]})
    returns = sa._calculate_daily_returns(prices).reset_index(drop=True)

    # Cálculo manual esperado:
    expected = pd.Series([(102-100)/100, (101-102)/102, (103-101)/101])

    assert len(returns) == len(expected)
    assert all(np.isclose(returns, expected, rtol=1e-5))

def test_calculate_daily_returns_empty():
    df = pd.DataFrame({"close": []})
    returns = sa._calculate_daily_returns(df)
    assert returns.empty


def test_calculate_daily_returns_missing_column():
    df = pd.DataFrame({"open": [1, 2, 3]})
    returns = sa._calculate_daily_returns(df)
    assert returns.empty


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

def test_perform_anova_analysis_insufficient_data(setup_folder):
    mock_data = {
        "BTC_USDT": pd.DataFrame({"close": []}),
        "ETH_USDT": pd.DataFrame({"close": []}),
    }

    sa.perform_anova_analysis(all_data=mock_data, save_folder=setup_folder)
    assert not os.path.exists(os.path.join(setup_folder, "anova_report_all_cryptos.txt"))
