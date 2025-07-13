# -*- coding: utf-8 -*-
import logging
from typing import List, Optional
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression


"""
Pré-processamento de Features com VIF, StandardScaler e SelectKBest.

Este módulo filtra, padroniza e seleciona variáveis com objetivo de reduzir
overfitting e melhorar a robustez dos modelos de regressão e redes neurais.

Explicação das etapas:
1. Remoção de colinearidade com VIF
Algumas variáveis estavam dizendo praticamente a mesma coisa. Por exemplo: a média móvel de 7 dias e a de 14 dias são parecidas. Quando temos muitas dessas variáveis juntas, os modelos podem se confundir e amplificar ruídos.
Usamos um método chamado VIF (Variance Inflation Factor) que mede se uma variável está "repetindo" as outras. Se o VIF for maior que 5, a gente exclui essa variável.

2. Padronização com StandardScaler
Algumas variáveis tinham valores gigantes (como volume de transações), enquanto outras tinham valores pequenos (como o RSI). Isso desequilibra os modelos, principalmente redes neurais e regressão.
A solução é escalar tudo para a mesma base — com média zero e desvio padrão 1 — usando o StandardScaler.

3. Seleção das melhores variáveis com SelectKBest
Depois de limpar e padronizar, ainda tínhamos muitas variáveis. Nem todas ajudam de verdade a prever o preço.
Usamos um método chamado f_regression, que calcula quais variáveis têm relação estatística com o preço (close) — e escolhemos as 10 melhores (k=10).
Isso ajuda a deixar o modelo mais leve, rápido e menos propenso a overfitting.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore


def remove_high_vif_features(X: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Remove features com VIF acima do limiar especificado.

    Args:
        X (pd.DataFrame): DataFrame com features.
        threshold (float): Valor de corte para o VIF.

    Returns:
        pd.DataFrame: DataFrame com apenas features com VIF aceitável.
    """
    X = X.copy()  # type: ignore
    while True:
        X = X.select_dtypes(include=[np.number]).astype(np.float64)  # type: ignore
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns,
        )
        max_vif = vif.max()
        if max_vif > threshold:
            drop_feature = vif.idxmax()
            print(f"[VIF] Removendo '{drop_feature}' com VIF={max_vif:.2f}")
            X = X.drop(columns=drop_feature)  # type: ignore
        else:
            break
    return X


def preprocess_features(
    X: pd.DataFrame,
    y: pd.Series,  # type: ignore
    vif_threshold: float = 10.0,
    k_best: int = 10,
    force_include: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aplica um pipeline de seleção de features para reduzir overfitting e melhorar desempenho do modelo.

    Este pipeline inclui:
    - Remoção de colunas com alta multicolinearidade via VIF (Variance Inflation Factor).
    - Seleção das K melhores variáveis com base no teste F (SelectKBest).
    - Inclusão forçada de colunas específicas mesmo que tenham sido removidas por critérios automáticos.

    Args:
        X (pd.DataFrame): Conjunto de variáveis independentes (features).
        y (pd.Series): Variável dependente (target), usada para cálculo de score no SelectKBest.
        vif_threshold (float, optional): Valor limite de VIF acima do qual a variável será removida. Padrão: 10.0.
        k_best (int, optional): Quantidade máxima de variáveis a manter após o SelectKBest. Padrão: 10.
        force_include (List[str], optional): Lista de variáveis que devem ser mantidas mesmo se descartadas.

    Returns:
        pd.DataFrame: Subconjunto do DataFrame X com as variáveis selecionadas.
    """
    # Remove multicolinearidade com base no VIF
    X_vif = remove_high_vif_features(X, threshold=vif_threshold)

    # Padroniza os dados com StandardScaler (após VIF)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_vif), columns=X_vif.columns, index=X_vif.index  # type: ignore
    )

    # Aplica SelectKBest para selecionar as k melhores variáveis
    k_best = min(k_best, X_vif.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k_best)
    selector.fit(X_scaled, y)  # type: ignore
    selected_columns = X_scaled.columns[selector.get_support()].tolist()

    # Garante que variáveis obrigatórias (ex: usd_brl) sejam mantidas
    if force_include:
        for col in force_include:
            if col in X.columns and col not in selected_columns:
                logging.info(
                    f"[Preprocessing] Forçando inclusão de '{col}' nas features."
                )
                # Padroniza individualmente e adiciona à matriz
                scaler_single = StandardScaler()
                X_scaled[col] = scaler_single.fit_transform(X[[col]])  # type: ignore
                selected_columns.append(col)

    logging.info(f"[SelectKBest] Features selecionadas: {selected_columns}")
    return X_scaled[selected_columns].copy()
