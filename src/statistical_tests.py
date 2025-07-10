# -*- coding: utf-8 -*-
"""
Módulo de Análise Estatística para Dados Financeiros.

Este módulo fornece um conjunto de ferramentas para realizar testes estatísticos
formais em dados de séries temporais financeiras, com foco em retornos de
criptoativos. Ele foi projetado para validar hipóteses sobre o comportamento
dos ativos e comparar o desempenho entre diferentes grupos.

Funcionalidades Principais:
-   **Teste de Hipótese (Teste t de 1 Amostra):** Permite testar se o retorno
    médio de um ativo é estatisticamente superior a um valor de referência.
-   **Análise de Variância (ANOVA):** Compara os retornos médios entre múltiplos
    ativos para determinar se existem diferenças significativas entre eles.
    A análise é realizada de duas formas:
    1.  Comparando diretamente os retornos de cada criptomoeda.
    2.  Agrupando as criptomoedas por nível de volatilidade (baixa, média, alta)
        e comparando os retornos médios desses grupos.
-   **Teste Post-Hoc de Tukey HSD:** Quando a ANOVA indica uma diferença
    significativa, este teste é executado automaticamente para identificar
    quais pares de grupos específicos diferem entre si.
-   **Geração de Relatórios:** Todas as análises geram relatórios detalhados em
    arquivos de texto e gráficos visuais (para o teste de Tukey), que são
    salvos em disco para fácil consulta e documentação.
"""
import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Configuração do Logging ---
# Configura um logging básico para exibir mensagens no console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Funções Auxiliares ---

def _calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    """
    Calcula os retornos diários a partir da coluna 'close' de um DataFrame.

    Esta função auxiliar interna isola e padroniza o cálculo dos retornos,
    evitando a modificação dos DataFrames originais (efeitos secundários).

    Args:
        df (pd.DataFrame): DataFrame que deve conter a coluna 'close'.

    Returns:
        pd.Series: Uma série com os retornos diários, já sem o primeiro valor
                   que é sempre NaN. Retorna uma série vazia se os dados de
                   entrada forem inválidos.
    """
    if 'close' not in df.columns or df['close'].isnull().all():
        return pd.Series(dtype=np.float64)
    # .pct_change() calcula a variação percentual.
    # .dropna() remove o primeiro valor que será NaN.
    return df['close'].pct_change().dropna()

# --- Funções de Testes Estatísticos ---

def perform_hypothesis_test(
    df: pd.DataFrame,
    pair_name: str,
    target_return_percent: float,
    save_folder: str,
    alpha: float = 0.05
):
    """
    Realiza um teste de hipótese para verificar se o retorno esperado médio
    é superior a um valor alvo (x%).

    Hipótese Nula (H0): O retorno médio diário é menor ou igual a x%.
    Hipótese Alternativa (H1): O retorno médio diário é maior que x%.

    Args:
        df (pd.DataFrame): DataFrame com os dados da criptomoeda.
        pair_name (str): Nome do par de criptomoedas (ex: 'BTC_USDT').
        target_return_percent (float): O valor x% (em formato decimal) a ser testado.
        save_folder (str): Pasta para salvar os relatórios.
        alpha (float): Nível de significância (padrão 0.05).
    """
    logging.info(f"Iniciando teste de hipótese para {pair_name}...")

    daily_returns = _calculate_daily_returns(df)

    if daily_returns.empty:
        logging.warning(f"Não foi possível calcular retornos diários para {pair_name}. Teste de hipótese cancelado.")
        return

    sample_mean = daily_returns.mean()
    sample_std = daily_returns.std()
    n = len(daily_returns)

    # Realiza o teste t de uma amostra.
    # 'alternative="greater"' especifica que é um teste unilateral à direita (H1: media > popmean).
    t_statistic, p_value = stats.ttest_1samp(
        daily_returns,
        popmean=target_return_percent,
        alternative='greater'
    )

    logging.info(f"  --- Resultados do Teste de Hipótese para {pair_name} ---")
    logging.info(f"  Retorno Médio da Amostra: {sample_mean:.6f}")
    logging.info(f"  Retorno Alvo (H0): {target_return_percent:.6f}")
    logging.info(f"  Estatística t: {t_statistic:.4f}")
    logging.info(f"  P-valor: {p_value:.4f}")

    if p_value < alpha:
        conclusion = (f"Rejeitamos a hipótese nula. Há evidências estatísticas para afirmar que o "
                      f"retorno médio diário de {pair_name} é SUPERIOR a {target_return_percent*100:.2f}%.")
    else:
        conclusion = (f"Não rejeitamos a hipótese nula. Não há evidências estatísticas para afirmar que o "
                      f"retorno médio diário de {pair_name} é SUPERIOR a {target_return_percent*100:.2f}%.")
    logging.info(f"  Conclusão: {conclusion}")

    # Salva os resultados em um arquivo de texto.
    report_path = os.path.join(save_folder, f"hypothesis_test_report_{pair_name}.txt")
    os.makedirs(save_folder, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Relatório de Teste de Hipótese para {pair_name}\n")
        f.write("-" * 50 + "\n")
        f.write(f"H0: Retorno médio diário <= {target_return_percent*100:.2f}%\n")
        f.write(f"H1: Retorno médio diário > {target_return_percent*100:.2f}%\n")
        f.write(f"Nível de Significância (alpha): {alpha}\n\n")
        f.write(f"Retorno Médio da Amostra: {sample_mean:.6f}\n")
        f.write(f"Desvio Padrão da Amostra: {sample_std:.6f}\n")
        f.write(f"Tamanho da Amostra (n): {n}\n")
        f.write(f"Estatística t: {t_statistic:.4f}\n")
        f.write(f"P-valor: {p_value:.4f}\n\n")
        f.write(f"Conclusão: {conclusion}\n")
    logging.info(f"Relatório salvo em: {report_path}")


def perform_anova_analysis(
    all_data: Dict[str, pd.DataFrame],
    save_folder: str,
    alpha: float = 0.05
):
    """
    Realiza Análise de Variância (ANOVA) para comparar os retornos médios.

    Esta função executa duas análises ANOVA distintas:
    1.  Compara os retornos médios diários entre todas as criptomoedas fornecidas.
    2.  Agrupa as criptomoedas por volatilidade (baixa, média, alta) e compara
        os retornos médios entre esses grupos.

    Para cada análise, se a ANOVA for significativa, um teste post-hoc de Tukey
    é realizado para identificar quais pares de grupos diferem.

    Args:
        all_data (Dict[str, pd.DataFrame]): Dicionário com nomes de criptomoedas
                                            como chaves e seus DataFrames como valores.
        save_folder (str): Pasta para salvar os relatórios e gráficos.
        alpha (float, optional): Nível de significância. Padrão 0.05.

    Side Effects:
        - Salva relatórios (.txt) e gráficos (.png) no `save_folder`.
    """
    logging.info("Iniciando análise ANOVA...")
    os.makedirs(save_folder, exist_ok=True)

    # 1. Preparação dos Dados
    all_returns = {}
    for name, df in all_data.items():
        daily_returns = _calculate_daily_returns(df)
        if not daily_returns.empty:
            clean_name = name.replace('_USDT', '')
            all_returns[clean_name] = daily_returns
        else:
            logging.warning(f"Ignorando {name} na análise ANOVA por falta de dados de retorno válidos.")

    if len(all_returns) < 2:
        logging.error("São necessárias pelo menos duas criptomoedas com dados válidos para realizar ANOVA.")
        return

    # 2. ANOVA entre todas as criptomoedas
    _run_anova_and_tukey(
        data_groups=all_returns,
        group_type_name="Criptomoeda",
        report_filename="anova_report_all_cryptos.txt",
        plot_filename="tukey_hsd_all_cryptos.png",
        save_folder=save_folder,
        alpha=alpha
    )

    # 3. ANOVA por agrupamento de volatilidade
    logging.info("Realizando ANOVA para comparar retornos entre grupos de volatilidade...")

    volatilities = pd.Series({name: returns.std() for name, returns in all_returns.items()})
    
    if len(volatilities) < 3:
        logging.warning("Dados insuficientes para agrupar por volatilidade. São necessários no mínimo 3 ativos.")
        return

    # Agrupamento em "baixa", "média", "alta" volatilidade
    low_thresh = volatilities.quantile(0.33)
    high_thresh = volatilities.quantile(0.66)

    volatility_groups = {
        'Baixa Volatilidade': [all_returns[name] for name, vol in volatilities.items() if vol <= low_thresh],
        'Média Volatilidade': [all_returns[name] for name, vol in volatilities.items() if low_thresh < vol <= high_thresh],
        'Alta Volatilidade': [all_returns[name] for name, vol in volatilities.items() if vol > high_thresh]
    }
    
    # Filtra grupos que possam ter ficado vazios
    volatility_groups_for_anova = {k: v for k, v in volatility_groups.items() if v}

    if len(volatility_groups_for_anova) < 2:
        logging.warning("Menos de 2 grupos de volatilidade formados. ANOVA por grupo cancelada.")
        return
        
    # Concatena os retornos de cada cripto dentro de seu grupo de volatilidade
    final_groups = {name: pd.concat(returns_list, ignore_index=True) for name, returns_list in volatility_groups_for_anova.items()}

    _run_anova_and_tukey(
        data_groups=final_groups,
        group_type_name="Grupo de Volatilidade",
        report_filename="anova_report_volatility_groups.txt",
        plot_filename="tukey_hsd_volatility_groups.png",
        save_folder=save_folder,
        alpha=alpha
    )

def _run_anova_and_tukey(
    data_groups: Dict[str, pd.Series],
    group_type_name: str,
    report_filename: str,
    plot_filename: str,
    save_folder: str,
    alpha: float
):
    """
    Função auxiliar para executar o ciclo ANOVA -> Tukey HSD -> Relatório.
    """
    logging.info(f"Executando ANOVA para grupos de '{group_type_name}'...")
    
    group_names = list(data_groups.keys())
    returns_list = list(data_groups.values())

    f_statistic, p_value = stats.f_oneway(*returns_list)

    logging.info(f"  --- ANOVA entre {group_type_name}s ---")
    logging.info(f"  Grupos Analisados: {group_names}")
    logging.info(f"  F-Estatística: {f_statistic:.4f}")
    logging.info(f"  P-valor: {p_value:.4f}")

    if p_value < alpha:
        conclusion = f"Rejeitamos a hipótese nula. Há uma diferença significativa entre os retornos médios dos {group_type_name}s."
        logging.info(f"  Conclusão: {conclusion}")
        
        # Prepara dados para o teste post-hoc de Tukey
        tukey_data = pd.concat(
            [pd.DataFrame({'returns': returns, 'group': name}) for name, returns in data_groups.items()],
            ignore_index=True
        )
        
        tukey_result = pairwise_tukeyhsd(endog=tukey_data['returns'], groups=tukey_data['group'], alpha=alpha)
        logging.info("Resultados do teste Post Hoc (Tukey HSD):\n" + str(tukey_result))

        # Plota os resultados do Tukey HSD
        fig = tukey_result.plot_simultaneous()
        plt.title(f'Teste Post Hoc de Tukey HSD - {group_type_name}s')
        plt.tight_layout()
        plot_path = os.path.join(save_folder, plot_filename)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logging.info(f"Gráfico Tukey HSD salvo em: {plot_path}")
        
    else:
        conclusion = f"Não rejeitamos a hipótese nula. Não há diferença significativa entre os retornos médios dos {group_type_name}s."
        tukey_result = None
        logging.info(f"  Conclusão: {conclusion}")

    # Salva o relatório da ANOVA
    report_path = os.path.join(save_folder, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Relatório de Análise de Variância (ANOVA) - {group_type_name}s\n")
        f.write("-" * 70 + "\n")
        f.write(f"Grupos Analisados: {', '.join(group_names)}\n")
        f.write(f"H0: Os retornos médios são iguais entre os grupos.\n")
        f.write(f"H1: Pelo menos um retorno médio difere.\n")
        f.write(f"Nível de Significância (alpha): {alpha}\n\n")
        f.write(f"F-Estatística: {f_statistic:.4f}\n")
        f.write(f"P-valor: {p_value:.4f}\n\n")
        f.write(f"Conclusão: {conclusion}\n")
        if tukey_result:
            f.write("\nResultados do Teste Post Hoc (Tukey HSD):\n")
            f.write(str(tukey_result))
    logging.info(f"Relatório ANOVA salvo em: {report_path}")


# --- Bloco de Exemplo de Execução ---
if __name__ == '__main__':
    logging.info("Executando script em modo de exemplo...")

    # Cria uma pasta para salvar os resultados
    results_folder = "statistical_reports"
    os.makedirs(results_folder, exist_ok=True)

    # 1. Cria dados de exemplo (mock data)
    np.random.seed(42) # Para resultados reproduzíveis
    days = 252 # Aproximadamente um ano de dias de negociação
    
    # Criando 3 ativos com médias de retorno e volatilidades diferentes
    btc_returns = np.random.normal(loc=0.001, scale=0.02, size=days) # Média maior, vol. média
    eth_returns = np.random.normal(loc=0.0005, scale=0.025, size=days) # Média menor, vol. maior
    ada_returns = np.random.normal(loc=0.0011, scale=0.015, size=days) # Média maior, vol. baixa

    # Converte retornos para preços (começando de um preço base 100)
    def returns_to_prices(returns, initial_price=100):
        return initial_price * (1 + returns).cumprod()

    mock_data = {
        "BTC_USDT": pd.DataFrame({'close': returns_to_prices(btc_returns, 50000)}),
        "ETH_USDT": pd.DataFrame({'close': returns_to_prices(eth_returns, 3000)}),
        "ADA_USDT": pd.DataFrame({'close': returns_to_prices(ada_returns, 1.5)}),
        "SOL_USDT": pd.DataFrame({'close': returns_to_prices(np.random.normal(0.002, 0.03, days), 150)}),
        "XRP_USDT": pd.DataFrame({'close': returns_to_prices(np.random.normal(0.0001, 0.022, days), 0.8)}),
        "EMPTY_DF": pd.DataFrame({'close': []}) # Exemplo de dado inválido
    }

    # 2. Executa o teste de hipótese para um dos ativos
    # H0: Retorno médio do BTC <= 0.05%
    # H1: Retorno médio do BTC > 0.05%
    perform_hypothesis_test(
        df=mock_data["BTC_USDT"],
        pair_name="BTC_USDT",
        target_return_percent=0.0005, # 0.05%
        save_folder=results_folder,
        alpha=0.05
    )

    logging.info("\n" + "="*50 + "\n")

    # 3. Executa a análise ANOVA completa
    perform_anova_analysis(
        all_data=mock_data,
        save_folder=results_folder,
        alpha=0.05
    )
    
    logging.info(f"\nAnálises concluídas. Verifique a pasta '{results_folder}' para os relatórios.")
