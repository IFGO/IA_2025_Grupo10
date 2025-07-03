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
    Simula um investimento de $1,000.00 e calcula o lucro obtido por diferentes modelos.
    Plota a evolução do lucro para cada modelo.

    Args:
        X (pd.DataFrame): Features de entrada.
        y (pd.Series): Variável alvo (preço de fechamento real).
        dates (pd.Series): Datas correspondentes aos dados.
        pair_name (str): Nome do par de criptomoedas.
        models_folder (str): Pasta onde os modelos treinados estão salvos.
        profit_plots_folder (str): Pasta para salvar os gráficos de lucro.
        initial_investment (float): Valor inicial do investimento.
    """
    logging.info(f"Simulando investimento e lucro para {pair_name}...")

    # Garante que X, y e dates têm os mesmos índices e estão alinhados
    data_df = pd.DataFrame({'date': dates, 'close': y}).set_index(X.index)
    data_df = pd.concat([data_df, X], axis=1)
    data_df = data_df.sort_values('date').reset_index(drop=True)

    # Carrega os modelos treinados
    model_types = ['mlp', 'linear', 'polynomial', 'randomforest']
    loaded_models = {}
    for m_type in model_types:
        model_filename = os.path.join(models_folder, f"{m_type}_{pair_name.replace(' ', '_')}.pkl")
        if os.path.exists(model_filename):
            try:
                loaded_models[m_type] = joblib.load(model_filename)
                logging.info(f"Modelo {m_type} carregado com sucesso para {pair_name}.")
            except Exception as e:
                logging.error(f"Falha ao carregar o modelo {m_type} para {pair_name}: {e}")
        else:
            logging.warning(f"Modelo {m_type} não encontrado para {pair_name} em {model_filename}. Ignorando.")

    if not loaded_models:
        logging.error(f"Nenhum modelo encontrado para simular lucro para {pair_name}. Treine os modelos primeiro.")
        return

    # Alinhar o DataFrame de datas já no início, para todos os modelos
    profit_evolution = pd.DataFrame({'date': data_df['date'].iloc[1:].reset_index(drop=True)})

    for model_key, model in loaded_models.items():
        logging.info(f"Executando simulação para o modelo: {model_key}")
        current_balance = initial_investment
        daily_balances = [initial_investment]

        # Para cada dia, exceto o último (precisamos do preço do dia seguinte para previsão)
        for i in range(len(data_df) - 1):
            # Features para o dia atual (para prever o próximo dia)
            current_day_features = data_df.iloc[i][X.columns].values.reshape(1, -1)

            # Previsão do preço de fechamento do próximo dia
            try:
                predicted_next_close = model.predict(current_day_features)[0]
            except Exception as e:
                logging.warning(f"Erro na previsão do modelo {model_key} no dia {data_df['date'].iloc[i]}: {e}. Usando preço atual.")
                predicted_next_close = data_df['close'].iloc[i] # Fallback

            current_close = data_df['close'].iloc[i]
            next_day_close = data_df['close'].iloc[i+1] # Preço real do próximo dia

            # Lógica de investimento: se a previsão do próximo dia for superior ao dia atual
            if predicted_next_close > current_close:
                # Investe todo o saldo acumulado
                # Calcula o retorno percentual do próximo dia
                daily_return_actual = (next_day_close - current_close) / current_close
                current_balance *= (1 + daily_return_actual)
            # Se a previsão for menor ou igual, não investe (mantém o saldo atual)

            daily_balances.append(current_balance)

        # Adiciona a coluna de saldo para o modelo atual
        profit_evolution[f'balance_{model_key}'] = daily_balances[1:]

    # Plotar o gráfico de evolução do lucro (Requisito 9f)
    plt.figure(figsize=(16, 9))
    sns.set_palette("tab10")

    for model_key in loaded_models.keys():
        col_name = f'balance_{model_key}'
        if col_name in profit_evolution.columns:
            plt.plot(profit_evolution['date'], profit_evolution[col_name], label=f'Modelo: {model_key.upper()}')

    plt.title(f'Evolução do Lucro com Investimento de $1,000.00 - {pair_name}', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Saldo Acumulado (USDT)', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plot_path = os.path.join(profit_plots_folder, f"profit_evolution_{pair_name.replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"Gráfico de evolução do lucro salvo em: {plot_path}")
