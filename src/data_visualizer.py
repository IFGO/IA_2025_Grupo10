import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

def plot_crypto_data(df: pd.DataFrame, pair_name: str, save_folder: str):
    """
    Gera e salva um gráfico de linha do preço de fecho de uma criptomoeda.

    Args:
        df (pd.DataFrame): O DataFrame que contém os dados da criptomoeda.
                           Deve incluir as colunas 'date' e 'close'.
        pair_name (str): O nome do par de moedas para ser usado no título do
                         gráfico (ex: "BTC_USDT").
        save_folder (str): O caminho da pasta onde a imagem do gráfico
                           será guardada.
    """
    try:
        logging.info(f"Gerando gráfico simples para {pair_name}...")

        if 'date' not in df.columns or 'close' not in df.columns:
            logging.error(f"DataFrame para {pair_name} não contém as colunas 'date' ou 'close'.")
            return

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['date', 'close'], inplace=True) 

        if df.empty:
            logging.warning(f"DataFrame vazio ou sem dados válidos para plotar para {pair_name}.")
            return

        df = df.sort_values('date')

        short_window = 20
        long_window = 50
        df['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1).mean()
        df['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1).mean()
        
        if 'signal' not in df.columns:
            previous_short_mavg = np.roll(df['short_mavg'], 1)
            previous_long_mavg = np.roll(df['long_mavg'], 1)
            df['signal'] = np.where(
                (df['short_mavg'] > df['long_mavg']) & (previous_short_mavg <= previous_long_mavg), 1,
                np.where((df['short_mavg'] < df['long_mavg']) & (previous_short_mavg >= previous_long_mavg), -1, 0)
            )
        plt.figure(figsize=(15, 8))
        plt.plot(df['date'], df['close'], label='Preço de Fechamento', color='skyblue', linewidth=1.5, alpha=0.8)
        plt.plot(df['date'], df['short_mavg'], label=f'Média Móvel ({short_window} dias)', color='orange', linestyle='--', linewidth=1)
        plt.plot(df['date'], df['long_mavg'], label=f'Média Móvel ({long_window} dias)', color='purple', linestyle='--', linewidth=1)

        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]

        plt.scatter(buy_signals['date'], buy_signals['close'], label='Sinal de Compra', marker='^', color='green', s=100, zorder=5)
        plt.scatter(sell_signals['date'], sell_signals['close'], label='Sinal de Venda', marker='v', color='red', s=100, zorder=5)


        plt.title(f'Histórico de Preço - {pair_name}', fontsize=16)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Preço de Fechamento (USDT)', fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.gcf().autofmt_xdate()

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plot_filename = os.path.join(save_folder, f"{pair_name.replace(' ', '_')}_chart.png")
        plt.savefig(plot_filename, dpi=150) 
        plt.close() 

        logging.info(f"Gráfico simples salvo em: {plot_filename}")

    except Exception as e:
        logging.error(f"Falha ao gerar gráfico para {pair_name}. Erro: {e}")

