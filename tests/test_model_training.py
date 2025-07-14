import pytest
import pandas as pd
import os
import logging

from sklearn.datasets import make_regression
from src.model_training import (
    train_and_evaluate_model,  # type: ignore
    compare_models,  # type: ignore
    get_best_model_by_mse,  # type: ignore
    limpar_modelos_antigos,
)

"""
Cria e retorna um conjunto de dados sintético para testes, com 50 amostras e 3 features,
utilizando make_regression do sklearn. A saída é uma tupla contendo um DataFrame com
as features e uma Series com os rótulos (target).
"""


@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)  # type: ignore
    df_X = pd.DataFrame(X, columns=["feat1", "feat2", "feat3"])
    series_y = pd.Series(y)
    return df_X, series_y


"""
Cria um diretório temporário chamado models para armazenar
arquivos gerados durante os testes, usando o fixture tmp_path do pytest.
"""


@pytest.fixture
def temp_folder(tmp_path):  # type: ignore
    folder = tmp_path / "models"  # type: ignore
    folder.mkdir()  # type: ignore
    return folder  # type: ignore


"""
Testa o treinamento e a persistência de um modelo de regressão linear. Verifica se um arquivo
com prefixo linear_test_linear foi salvo corretamente no diretório temporário.
"""


def test_train_linear_model(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "Linear", kfolds=3, pair_name="test_linear", models_folder=str(temp_folder))  # type: ignore
    assert any(f.name.startswith("linear_test_linear") for f in temp_folder.iterdir())  # type: ignore


"""
Testa o treinamento de um modelo de regressão polinomial com grau válido (3). Verifica
se um arquivo com prefixo polynomial_test_poly foi gerado no diretório temporário.
"""


def test_train_polynomial_model_valid(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "Polynomial", kfolds=3, pair_name="test_poly", models_folder=str(temp_folder), poly_degree=3)  # type: ignore
    assert any(f.name.startswith("polynomial_test_poly") for f in temp_folder.iterdir())  # type: ignore


"""
Testa o comportamento ao treinar um modelo polinomial com grau inválido (1).
Espera-se que nenhum modelo seja salvo no diretório.
"""


def test_train_polynomial_model_invalid_degree(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "Polynomial", kfolds=3, pair_name="test_invalid", models_folder=str(temp_folder), poly_degree=1)  # type: ignore
    assert len(list(temp_folder.iterdir())) == 0  # type: ignore


"""
Testa o treinamento de um modelo do tipo MLP (Multi-Layer Perceptron). Verifica se um
arquivo com prefixo mlp_test_mlp foi salvo corretamente.
"""


def test_train_mlp_model(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "MLP", kfolds=3, pair_name="test_mlp", models_folder=str(temp_folder))  # type: ignore
    assert any(f.name.startswith("mlp_test_mlp") for f in temp_folder.iterdir())  # type: ignore


"""
Testa o treinamento de um modelo Random Forest. Verifica se um arquivo com prefixo
randomforest_test_rf foi salvo no diretório.
"""


def test_train_randomforest_model(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "RandomForest", kfolds=3, pair_name="test_rf", models_folder=str(temp_folder))  # type: ignore
    assert any(f.name.startswith("randomforest_test_rf") for f in temp_folder.iterdir())  # type: ignore


"""
Verifica se o sistema lida corretamente com um tipo de modelo inválido ("InvalidModel").
Espera-se que nenhum arquivo seja gerado.
"""


def test_invalid_model_type(sample_data, temp_folder):  # type: ignore
    X, y = sample_data  # type: ignore
    train_and_evaluate_model(X, y, "InvalidModel", kfolds=3, pair_name="test_invalid", models_folder=str(temp_folder))  # type: ignore
    assert len(list(temp_folder.iterdir())) == 0  # type: ignore


"""
Testa a função compare_models, que executa e compara múltiplos modelos.
Usa o caplog para verificar se a mensagem "Comparação de Modelos" foi registrada no log
durante a execução.
"""


def test_compare_models(sample_data, temp_folder, caplog):  # type: ignore
    X, y = sample_data  # type: ignore
    with caplog.at_level(logging.INFO):  # type: ignore
        compare_models(X, y, kfolds=3, pair_name="test_compare", plots_folder=str(temp_folder))  # type: ignore
    assert "Comparação de Modelos" in caplog.text  # type: ignore


"""
Testa a função get_best_model_by_mse, que retorna o modelo com menor erro quadrático médio.
Verifica se o modelo retornado não é None e se o nome está entre os modelos esperados.
"""


def test_get_best_model_by_mse(sample_data):  # type: ignore
    X, y = sample_data  # type: ignore
    model, name = get_best_model_by_mse(X, y, kfolds=3)  # type: ignore
    assert model is not None
    assert name in {"MLP", "Linear", "Polynomial", "RandomForest"}


"""
Testa a função limpar_modelos_antigos, que remove arquivos antigos de modelos com base
no nome do par de moedas. Cria arquivos falsos e verifica se todos são removidos
corretamente.
"""


def test_limpar_modelos_antigos(temp_folder):  # type: ignore
    # cria arquivos falsos
    nomes = ["mlp", "linear", "polynomial", "randomforest"]
    for nome in nomes:
        path = os.path.join(temp_folder, f"{nome}_BTC_USDT.pkl")  # type: ignore
        open(path, "w").close()

    limpar_modelos_antigos("BTC_USDT", str(temp_folder))  # type: ignore

    assert len(list(temp_folder.iterdir())) == 0  # type: ignore
