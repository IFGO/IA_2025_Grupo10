# -*- coding: utf-8 -*-

"""
Ferramentas de Análise de Dados de Criptoativos.

Este módulo fornece um conjunto de funções para realizar análises estatísticas
e visuais sobre dados históricos de preços de criptoativos. Ele utiliza as
bibliotecas Pandas para manipulação de dados, Matplotlib e Seaborn para a
geração de gráficos, e o módulo logging para registrar o progresso e possíveis
avisos.

As funcionalidades principais incluem:
- Cálculo de estatísticas descritivas detalhadas (média, desvio padrão,
  curtose, etc.).
- Geração de um painel de visualização consolidado, incluindo:
  - Histórico de preços em escala linear e logarítmica.
  - Histograma de distribuição de preços.
  - Boxplot para análise de quartis e detecção de outliers.
- Análise comparativa da volatilidade entre diferentes ativos, utilizando o
  Coeficiente de Variação para uma medição relativa.

As funções são projetadas para serem robustas, tratando casos de dados ausentes
ou inválidos e salvando os resultados gráficos em arquivos de imagem para fácil
consulta.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import logging
from typing import Dict

sns.set_theme(style="whitegrid")


def calculate_statistics(df: pd.DataFrame) -> pd.Series:  # type: ignore
    """Calcula um conjunto expandido de estatísticas descritivas para os preços.

    Esta função recebe um DataFrame e calcula as principais métricas estatísticas
    para a coluna 'close', incluindo média, desvio padrão, variância, quartis,
    assimetria (skewness) e curtose (kurtosis). As operações são vetorizadas
    pelo próprio Pandas para maior eficiência.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados históricos, que deve
                           incluir uma coluna 'close'.

    Returns:
        pd.Series: Uma série do Pandas contendo as estatísticas calculadas.
                   Retorna uma série vazia se o DataFrame for inválido.
    """
    if df.empty or "close" not in df.columns:
        logging.warning(
            "DataFrame vazio ou sem a coluna 'close'. Retornando estatísticas vazias."
        )
        return pd.Series(dtype=float)

    close_series = df["close"]  # type: ignore
    stats = close_series.describe()  # type: ignore
    stats["variance"] = close_series.var()
    stats["skewness"] = close_series.skew()
    stats["kurtosis"] = close_series.kurt()
    return stats  # type: ignore


def generate_analysis_plots(df: pd.DataFrame, pair_name: str, save_folder: str):
    """Gera e salva uma figura com múltiplos gráficos de análise para um ativo.

    Cria uma visualização consolidada que inclui:
    1. Gráfico de linha do histórico de preços com média, mediana e moda.
    2. Histograma da distribuição de preços.
    3. Boxplot para visualizar quartis e outliers.

    A imagem gerada é salva no diretório especificado.

    Args:
        df (pd.DataFrame): DataFrame com os dados do ativo, incluindo as colunas
                           'date' e 'close'.
        pair_name (str): O nome do par de moedas (ex: 'BTC/USDT') para ser
                         usado nos títulos dos gráficos.
        save_folder (str): O caminho da pasta onde o arquivo de imagem do
                           gráfico será salvo.
    """
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        logging.warning(
            f"Dados para '{pair_name}' estão vazios ou com colunas essenciais ('date', 'close') ausentes. Gráfico não gerado."
        )
        return

    logging.info(f"Gerando plots de análise para {pair_name}...")

    # Converte a coluna 'date' para datetime e remove linhas com datas inválidas
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # type: ignore
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)  # type: ignore

    fig = plt.figure(figsize=(20, 12))  # type: ignore
    gs = fig.add_gridspec(2, 2)  # type: ignore
    fig.suptitle(f"Análise Completa - {pair_name}", fontsize=20, y=0.95)  # type: ignore

    ax1 = fig.add_subplot(gs[0, :])  # type: ignore

    # Adiciona formatação elegante de datas
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax1.plot(df["date"], df["close"], label="Preço de Fechamento", color="navy", alpha=0.9)  # type: ignore

    mean_price = df["close"].mean()
    median_price = df["close"].median()
    mode_price = df["close"].mode()[0] if not df["close"].mode().empty else None  # type: ignore

    ax1.axhline(mean_price, color="red", linestyle="--", label=f"Média: ${mean_price:,.2f}")  # type: ignore
    ax1.axhline(median_price, color="green", linestyle="-.", label=f"Mediana: ${median_price:,.2f}")  # type: ignore
    if mode_price is not None:
        ax1.axhline(mode_price, color="purple", linestyle=":", label=f"Moda: ${mode_price:,.2f}")  # type: ignore

    ax1.set_title("Histórico de Preço de Fechamento com Métricas Centrais", fontsize=14)  # type: ignore
    ax1.set_ylabel("Preço (USDT)")  # type: ignore
    ax1.legend()  # type: ignore
    fig.autofmt_xdate()

    if not df.empty and (df["close"] > 0).all():
        ax1.set_yscale("log")  # type: ignore
        ax1.set_ylabel("Preço (USDT) - Escala Log")  # type: ignore
    else:
        logging.warning(
            f"Não foi possível aplicar escala logarítmica para '{pair_name}' devido a preços <= 0. Usando escala linear."
        )

    ax2 = fig.add_subplot(gs[1, 0])  # type: ignore
    sns.histplot(df["close"], kde=True, ax=ax2, color="skyblue")  # type: ignore
    ax2.set_title("Distribuição de Frequência (Histograma)")  # type: ignore
    ax2.set_xlabel("Preço de Fechamento")  # type: ignore
    ax2.set_ylabel("Frequência")  # type: ignore

    ax3 = fig.add_subplot(gs[1, 1])  # type: ignore
    sns.boxplot(x=df["close"], ax=ax3, color="lightgreen")  # type: ignore
    ax3.set_title("Distribuição (Boxplot)")  # type: ignore
    ax3.set_xlabel("Preço de Fechamento")  # type: ignore

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore

    os.makedirs(save_folder, exist_ok=True)

    safe_pair_name = pair_name.replace("/", "_").replace(" ", "_")
    plot_path = os.path.join(save_folder, f"analise_{safe_pair_name}.png")

    plt.savefig(plot_path, dpi=150)  # type: ignore
    plt.close(fig)
    logging.info(f"Gráfico de análise consolidado salvo em: {plot_path}")


def calculate_comparative_variability(
    all_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calcula e compara a variabilidade entre múltiplos ativos.

    Esta função mede a variabilidade relativa dos preços de fechamento de cada
    ativo utilizando o Coeficiente de Variação (CV). O CV é útil para comparar
    a volatilidade entre ativos com diferentes escalas de preço. A função opera
    de forma eficiente, evitando laços explícitos em favor de cálculos
    vetorizados com Pandas sempre que possível.

    Args:
        all_data (Dict[str, pd.DataFrame]): Um dicionário onde as chaves são os
                                            nomes dos ativos e os valores são
                                            os DataFrames correspondentes com
                                            os dados históricos.

    Returns:
        pd.DataFrame: Um DataFrame ordenado pelo Coeficiente de Variação
                      (do maior para o menor), contendo o preço médio, o desvio
                      padrão (variabilidade absoluta) e o CV (variabilidade
                      relativa) para cada ativo. Retorna um DataFrame vazio se
                      nenhum dado válido for encontrado.
    """
    results = []
    for name, df in all_data.items():
        if not df.empty and "close" in df.columns:
            mean = df["close"].mean()
            std_dev = df["close"].std()
            cv = (std_dev / mean) * 100 if mean != 0 else 0
            results.append(  # type: ignore
                {  # type: ignore
                    "Criptomoeda": name.replace("_", " "),
                    "Preço Médio": mean,
                    "Desvio Padrão (Variabilidade Absoluta)": std_dev,
                    "Coef. de Variação (%) (Variabilidade Relativa)": cv,
                }
            )
        else:
            logging.warning(
                f"Dados vazios ou sem coluna 'close' para {name}. Ignorando na análise de variabilidade."
            )

    if not results:
        logging.warning("Nenhum dado válido para calcular a variabilidade comparativa.")
        return pd.DataFrame(
            columns=[
                "Criptomoeda",
                "Preço Médio",
                "Desvio Padrão (Variabilidade Absoluta)",
                "Coef. de Variação (%) (Variabilidade Relativa)",
            ]
        )

    df_variability = pd.DataFrame(results)
    return df_variability.sort_values(by="Coef. de Variação (%) (Variabilidade Relativa)", ascending=False)  # type: ignore
