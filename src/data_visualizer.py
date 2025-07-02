# data_visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

def plot_crypto_data(csv_path: str, save_folder: str):
    """
    Lê um arquivo CSV de criptomoeda e gera um gráfico do preço de fechamento.

    Args:
        csv_path (str): O caminho para o arquivo CSV de entrada.
        save_folder (str): A pasta onde o gráfico de imagem será salvo.
    """
    try:
        pair_name = os.path.basename(csv_path).replace('.csv', '').replace('_', ' ').upper()
        logging.info(f"Gerando gráfico para {pair_name}...")
        
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        plt.figure(figsize=(14, 7))
        plt.plot(df['date'], df['close'], label='Preço de Fechamento (Close)')
        
        plt.title(f'Histórico de Preço - {pair_name}', fontsize=16)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Preço de Fechamento (USDT)', fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.gcf().autofmt_xdate()

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        plot_filename = os.path.join(save_folder, f"{pair_name.replace(' ', '_')}_chart.png")
        plt.savefig(plot_filename)
        plt.close() # Libera memória

        logging.info(f"Gráfico salvo em: {plot_filename}")

    except Exception as e:
        logging.error(f"Falha ao gerar gráfico para {csv_path}. Erro: {e}")