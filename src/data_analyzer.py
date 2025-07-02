# data_analyzer.py (Versão Revisada)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict

# Configura o estilo dos gráficos
sns.set_theme(style="whitegrid")

def calculate_statistics(df: pd.DataFrame) -> pd.Series:
    """
    Calcula um conjunto expandido de estatísticas para a coluna 'close'.
    Inclui média, desvio padrão, variância, quartis, min, max, assimetria e curtose.
    """
    close_series = df['close']
    
    # Usa describe() para as estatísticas base
    stats = close_series.describe()
    
    # Adiciona as estatísticas faltantes
    stats['variance'] = close_series.var()
    stats['skewness'] = close_series.skew()
    stats['kurtosis'] = close_series.kurt()
    
    return stats

def generate_analysis_plots(df: pd.DataFrame, pair_name: str, save_folder: str):
    """
    Gera uma figura única com múltiplos gráficos de análise:
    1. Gráfico de linha do preço de fechamento com média, mediana e moda.
    2. Histograma da distribuição dos preços.
    3. Boxplot para visualizar os quartis e outliers.
    """
    logging.info(f"Gerando plots de análise para {pair_name}...")

    # Cria uma figura com 2 linhas e 2 colunas para os gráficos
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)
    
    # Título principal da figura
    fig.suptitle(f'Análise Completa - {pair_name}', fontsize=20, y=0.95)

    # --- Gráfico 1: Série Temporal (ocupa a linha superior inteira) ---
    ax1 = fig.add_subplot(gs[0, :]) # Ocupa a primeira linha, todas as colunas
    ax1.plot(df['date'], df['close'], label='Preço de Fechamento', color='navy', alpha=0.9)
    
    mean_price = df['close'].mean()
    median_price = df['close'].median()
    mode_price = df['close'].mode()[0]
    
    ax1.axhline(mean_price, color='red', linestyle='--', label=f'Média: ${mean_price:,.2f}')
    ax1.axhline(median_price, color='green', linestyle='-.', label=f'Mediana: ${median_price:,.2f}')
    
    ax1.set_title('Histórico de Preço de Fechamento com Métricas Centrais', fontsize=14)
    ax1.set_ylabel('Preço (USDT) - Escala Log')
    ax1.set_yscale('log') # Escala logarítmica é essencial para cripto
    ax1.legend()
    fig.autofmt_xdate()

    # --- Gráfico 2: Histograma ---
    ax2 = fig.add_subplot(gs[1, 0]) # Linha 2, Coluna 1
    sns.histplot(df['close'], kde=True, ax=ax2, color='skyblue')
    ax2.set_title('Distribuição de Frequência (Histograma)')
    ax2.set_xlabel('Preço de Fechamento')

    # --- Gráfico 3: Boxplot ---
    ax3 = fig.add_subplot(gs[1, 1]) # Linha 2, Coluna 2
    sns.boxplot(x=df['close'], ax=ax3, color='lightgreen')
    ax3.set_title('Distribuição (Boxplot)')
    ax3.set_xlabel('Preço de Fechamento')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajusta o layout para caber o suptitle

    # Salva a figura consolidada
    plot_path = os.path.join(save_folder, f"analise_{pair_name.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()

def calculate_comparative_variability(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calcula e compara a variabilidade entre as criptomoedas usando o Coeficiente de Variação.
    """
    results = []
    for name, df in all_data.items():
        mean = df['close'].mean()
        std_dev = df['close'].std()
        cv = (std_dev / mean) * 100 if mean != 0 else 0
        results.append({
            "Criptomoeda": name.replace('_', ' '),
            "Preço Médio": mean,
            "Desvio Padrão (Variabilidade Absoluta)": std_dev,
            "Coef. de Variação (%) (Variabilidade Relativa)": cv
        })
    df_variability = pd.DataFrame(results)
    return df_variability.sort_values(by="Coef. de Variação (%) (Variabilidade Relativa)", ascending=False)