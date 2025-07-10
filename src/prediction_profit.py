# -*- coding: utf-8 -*-
"""
Simulação de Lucro e Backtesting de Estratégias de Investimento.

Este módulo é dedicado a simular o desempenho de uma estratégia de investimento
baseada nas previsões de modelos de machine learning. Ele utiliza uma abordagem
vetorizada para garantir alta performance ao calcular os retornos de uma
carteira ao longo do tempo.

Funcionalidades Principais:
-   **Carregamento de Modelos:** Carrega modelos de regressão pré-treinados
    (ex: MLP, Linear, RandomForest) a partir de arquivos.
-   **Geração de Sinais de Negociação:** Cria sinais de compra/venda com base
    em uma lógica simples: se o preço previsto para o dia seguinte for maior
    que o preço atual, um sinal de compra (manter posição) é gerado.
-   **Cálculo de Retorno Vetorizado:** Simula a evolução de um investimento
    inicial aplicando os sinais de negociação aos retornos diários do ativo,
    tudo de forma eficiente com Pandas e NumPy.
-   **Visualização de Desempenho:** Gera e salva um gráfico que compara a
    evolução do saldo (lucro/prejuízo) para cada um dos modelos testados,
    permitindo uma análise visual de qual modelo teria gerado o melhor retorno.
"""
import pandas as pd
import numpy as np
import logging
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def simulate_investment_and_profit(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    pair_name: str,
    models_folder: str,
    profit_plots_folder: str,
    initial_investment: float = 1000.0
):
    """
    Simula um investimento, calcula o lucro e plota a evolução do capital.

    Esta função realiza um backtest de uma estratégia de negociação simples. Ela
    carrega modelos pré-treinados, utiliza-os para prever os preços futuros e
    gera sinais de negociação com base nessas previsões. A lógica da estratégia é:
    manter o ativo (sinal=1) se o preço previsto for maior que o preço conhecido
    anterior; caso contrário, ficar fora do mercado (sinal=0).

    O cálculo da evolução do investimento é feito de forma vetorizada para máxima
    performance. Por fim, a função gera e salva um gráfico comparando o
    desempenho de cada modelo.

    Args:
        X (pd.DataFrame): DataFrame com as features de entrada para os modelos.
        y (pd.Series): Series com a variável alvo (preço de fecho real).
        dates (pd.Series): Series com as datas correspondentes aos dados.
        pair_name (str): Nome do par de moedas, usado para nomear arquivos.
        models_folder (str): Diretório onde os modelos treinados (.pkl) estão guardados.
        profit_plots_folder (str): Diretório para salvar os gráficos de lucro.
        initial_investment (float, optional): O valor inicial do investimento em USDT.
                                               Padrão é 1000.0.

    Side Effects:
        - Salva um gráfico (.png) da evolução do lucro no `profit_plots_folder`.
        - Registra o progresso da simulação e eventuais avisos no log.
    """
    logging.info(f"Simulando investimento e lucro de forma vetorizada para {pair_name}...")

    model_types = ['mlp', 'linear', 'polynomial', 'randomforest']
    loaded_models = {}
    for m_type in model_types:
        model_filename = os.path.join(models_folder, f"{m_type}_{pair_name.replace(' ', '_')}.pkl")
        if os.path.exists(model_filename):
            try:
                loaded_models[m_type] = joblib.load(model_filename)
            except Exception as e:
                logging.error(f"Falha ao carregar o modelo {m_type}: {e}")
        else:
            logging.warning(f"Modelo {m_type} não encontrado em {model_filename}. Ignorando.")

    if not loaded_models:
        logging.error(f"Nenhum modelo carregado para {pair_name}. Simulação cancelada.")
        return

    data_df = pd.DataFrame({'date': dates, 'close': y}).set_index(X.index)
    data_df = pd.concat([data_df, X], axis=1).sort_values('date').reset_index(drop=True)
    
    profit_evolution = pd.DataFrame({'date': data_df['date']})

    for model_key, model in loaded_models.items():
        logging.info(f"Executando simulação vetorizada para o modelo: {model_key}")
        
        all_predictions = model.predict(data_df[X.columns])
        
        last_known_price = data_df['close'].shift(1).fillna(0)
        signals = np.where(all_predictions > last_known_price, 1, 0)
        signals = pd.Series(signals, index=data_df.index).shift(1).fillna(0)

        daily_returns = data_df['close'].pct_change().fillna(0)
        
        strategy_returns = daily_returns * signals
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        profit_evolution[f'balance_{model_key}'] = initial_investment * cumulative_returns

    plt.figure(figsize=(16, 9))
    sns.set_palette("tab10")

    for model_key in loaded_models.keys():
        col_name = f'balance_{model_key}'
        if col_name in profit_evolution.columns:
            plt.plot(profit_evolution['date'], profit_evolution[col_name], label=f'Modelo: {model_key.upper()}')

    plt.title(f'Evolução do Lucro com Investimento de ${initial_investment:,.2f} - {pair_name}', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Saldo Acumulado (USDT)', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    if not os.path.exists(profit_plots_folder):
        os.makedirs(profit_plots_folder)
    plot_path = os.path.join(profit_plots_folder, f"profit_evolution_{pair_name.replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"Gráfico de evolução do lucro salvo em: {plot_path}")
