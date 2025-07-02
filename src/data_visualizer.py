# src/data_visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

def plot_crypto_data(df: pd.DataFrame, pair_name: str, save_folder: str):
    """
    Gera um gráfico de linha do preço de fechamento de um DataFrame de criptomoeda.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados da criptomoeda.
        pair_name (str): O nome do par de criptomoedas (ex: "BTC_USDT").
        save_folder (str): A pasta onde o gráfico de imagem será salvo.
    """
    try:
        logging.info(f"Gerando gráfico simples para {pair_name}...")

        # Ensure 'date' column is datetime and 'close' is numeric
        # This is already handled in data_loader, but good for robustness if df comes from elsewhere
        if 'date' not in df.columns or 'close' not in df.columns:
            logging.error(f"DataFrame para {pair_name} não contém as colunas 'date' ou 'close'.")
            return

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['date', 'close'], inplace=True) # Ensure no NaNs in critical columns

        if df.empty:
            logging.warning(f"DataFrame vazio ou sem dados válidos para plotar para {pair_name}.")
            return

        df = df.sort_values('date')

        plt.figure(figsize=(14, 7))
        plt.plot(df['date'], df['close'], label='Preço de Fechamento (Close)', color='blue')

        plt.title(f'Histórico de Preço - {pair_name}', fontsize=16)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Preço de Fechamento (USDT)', fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.gcf().autofmt_xdate() # Formata as datas no eixo X

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plot_filename = os.path.join(save_folder, f"{pair_name.replace(' ', '_')}_chart.png")
        plt.savefig(plot_filename, dpi=150) # Resolução mínima de 150 dpi
        plt.close() # Libera memória

        logging.info(f"Gráfico simples salvo em: {plot_filename}")

    except Exception as e:
        logging.error(f"Falha ao gerar gráfico para {pair_name}. Erro: {e}")

