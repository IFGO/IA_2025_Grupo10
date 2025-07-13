import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_features, remove_high_vif_features

"""
    Gera um conjunto de dados sintético para teste com 4 variáveis:
    - Três features independentes e uma colinear ("feature4"),
    - Target (y) correlacionado com "feature1".
"""
@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.normal(size=100),
        "feature2": np.random.normal(size=100),
        "feature3": np.random.normal(size=100)
    })
    X["feature4"] = X["feature1"] * 0.9 + np.random.normal(scale=0.1, size=100)  # Colinear
    y = X["feature1"] * 2 + np.random.normal(size=100)
    return X, y

"""
    Testa se a função `remove_high_vif_features` remove features com alta colinearidade (VIF alto).
    A feature "feature4" foi criada propositalmente colinear com "feature1".

    Verifica:
        - Se ao menos uma das colineares ("feature1" ou "feature4") foi removida.
        - Se o número de colunas resultante é menor que o original.
"""
def test_remove_high_vif_features_removes_colinear(sample_data):
    X, _ = sample_data
    X_filtered = remove_high_vif_features(X, threshold=5.0)
    assert "feature4" not in X_filtered.columns or "feature1" not in X_filtered.columns
    assert X_filtered.shape[1] < X.shape[1]

"""
    Testa se a função `remove_high_vif_features` não remove colunas quando não há colinearidade.

    Passos:
        - A feature colinear ("feature4") é removida previamente.
        - Verifica se nenhuma coluna adicional foi removida.
"""
def test_remove_high_vif_features_no_removal(sample_data):
    X, _ = sample_data
    X = X.drop(columns="feature4")  # Removendo colinear antes
    X_filtered = remove_high_vif_features(X, threshold=5.0)
    assert X_filtered.shape[1] == X.shape[1]

"""
    Testa a função `preprocess_features` com parâmetros básicos:
        - Remoção de colinearidade (via VIF).
        - Seleção das 2 melhores features com `SelectKBest`.

    Verifica:
        - Se a saída é um DataFrame.
        - Se o número final de colunas está de acordo com a filtragem aplicada.
"""
def test_preprocess_features_basic(sample_data):
    X, y = sample_data
    result = preprocess_features(X, y, vif_threshold=5.0, k_best=2)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] <= 3  # uma removida por VIF, duas escolhidas

"""
    Testa se a função `preprocess_features` respeita a inclusão forçada de colunas,
    mesmo que tenham VIF alto ou não estejam entre as melhores por `SelectKBest`.

    Verifica:
        - Se a coluna forçada ("feature4") está presente no resultado final.
"""
def test_preprocess_features_force_include(sample_data):
    X, y = sample_data
    col_forced = "feature4"
    result = preprocess_features(X, y, vif_threshold=5.0, k_best=2, force_include=[col_forced])
    assert col_forced in result.columns

"""
    Testa o comportamento da função `preprocess_features` ao solicitar mais features
    do que o número total disponível após filtragem por VIF.

    Verifica:
        - Se o número final de colunas não excede o número de colunas de entrada.
"""
def test_preprocess_features_kbest_limit(sample_data):
    X, y = sample_data
    result = preprocess_features(X, y, vif_threshold=5.0, k_best=20)  # maior que total de colunas
    assert result.shape[1] <= X.shape[1]
