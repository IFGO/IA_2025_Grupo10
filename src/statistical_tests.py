    # src/statistical_tests.py
    import pandas as pd
    import numpy as np
    import logging
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from typing import Dict
    from scipy import stats
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    def perform_hypothesis_test(
        df: pd.DataFrame,
        pair_name: str,
        target_return_percent: float,
        save_folder: str,
        alpha: float = 0.05
    ):
        """
        Realiza um teste de hipótese para verificar se o retorno esperado médio
        é superior ou igual a um valor definido pelo usuário (x%).

        H0: Retorno médio <= x%
        H1: Retorno médio > x%

        Args:
            df (pd.DataFrame): DataFrame com os dados da criptomoeda, incluindo 'close'.
            pair_name (str): Nome do par de criptomoedas.
            target_return_percent (float): O valor x% (em decimal) a ser testado.
            save_folder (str): Pasta para salvar os relatórios.
            alpha (float): Nível de significância (padrão 0.05).
        """
        logging.info(f"Realizando teste de hipótese para {pair_name}...")

        if df.empty or 'close' not in df.columns:
            logging.warning(f"Dados inválidos para teste de hipótese para {pair_name}.")
            return

        # Calcula os retornos diários
        df['daily_return'] = df['close'].pct_change().dropna()

        if df['daily_return'].empty:
            logging.warning(f"Não há retornos diários para {pair_name}. Impossível realizar o teste de hipótese.")
            return

        sample_mean = df['daily_return'].mean()
        sample_std = df['daily_return'].std()
        n = len(df['daily_return'])

        # Teste t de uma amostra (one-sample t-test)
        # H0: mu <= target_return_percent
        # H1: mu > target_return_percent (teste unilateral à direita)
        # stats.ttest_1samp retorna o p-valor para um teste bicaudal.
        # Para unilateral à direita, p_unilateral = p_bicaudal / 2 se t > 0, ou 1 - p_bicaudal / 2 se t < 0.
        # Ou simplesmente 1 - cdf(t) para t-statistic.

        t_statistic, p_value_two_sided = stats.ttest_1samp(df['daily_return'], popmean=target_return_percent, alternative='greater')

        # Para um teste unilateral à direita, o p-valor é diretamente dado por 'alternative='greater''
        p_value = p_value_two_sided

        logging.info(f"  --- Teste de Hipótese para {pair_name} ---")
        logging.info(f"  Retorno Médio da Amostra: {sample_mean:.6f}")
        logging.info(f"  Retorno Alvo (x%): {target_return_percent:.6f}")
        logging.info(f"  Estatística t: {t_statistic:.4f}")
        logging.info(f"  P-valor: {p_value:.4f}")
        logging.info(f"  Nível de Significância (alpha): {alpha}")

        conclusion = ""
        if p_value < alpha:
            conclusion = f"Rejeitamos a hipótese nula. Há evidências significativas para afirmar que o retorno médio diário de {pair_name} é SUPERIOR a {target_return_percent*100:.2f}%."
        else:
            conclusion = f"Não rejeitamos a hipótese nula. Não há evidências significativas para afirmar que o retorno médio diário de {pair_name} é SUPERIOR a {target_return_percent*100:.2f}%."
        logging.info(f"  Conclusão: {conclusion}")

        # Salvar resultados em um arquivo de texto
        report_path = os.path.join(save_folder, f"hypothesis_test_report_{pair_name.replace(' ', '_')}.txt")
        with open(report_path, 'w') as f:
            f.write(f"Relatório de Teste de Hipótese para {pair_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Hipótese Nula (H0): Retorno médio diário <= {target_return_percent*100:.2f}%\n")
            f.write(f"Hipótese Alternativa (H1): Retorno médio diário > {target_return_percent*100:.2f}%\n")
            f.write(f"Nível de Significância (alpha): {alpha}\n\n")
            f.write(f"Retorno Médio da Amostra: {sample_mean:.6f}\n")
            f.write(f"Desvio Padrão da Amostra: {sample_std:.6f}\n")
            f.write(f"Tamanho da Amostra (n): {n}\n")
            f.write(f"Estatística t: {t_statistic:.4f}\n")
            f.write(f"P-valor: {p_value:.4f}\n\n")
            f.write(f"Conclusão: {conclusion}\n")
        logging.info(f"Relatório de teste de hipótese salvo em: {report_path}")


    def perform_anova_analysis(
        all_data: Dict[str, pd.DataFrame],
        save_folder: str,
        alpha: float = 0.05
    ):
        """
        Realiza análises de variância (ANOVA) para comparar os retornos médios diários das criptomoedas.
        Inclui ANOVA entre todas as criptomoedas e entre grupos de criptomoedas.

        Args:
            all_data (Dict[str, pd.DataFrame]): Dicionário de DataFrames de criptomoedas.
            save_folder (str): Pasta para salvar os relatórios e gráficos.
            alpha (float): Nível de significância (padrão 0.05).
        """
        logging.info("Iniciando análise ANOVA...")

        if not all_data:
            logging.warning("Nenhum dado disponível para análise ANOVA.")
            return

        # --- 11a. ANOVA entre todas as criptomoedas ---
        logging.info("Realizando ANOVA para comparar retornos médios entre todas as criptomoedas...")
        returns_list = []
        crypto_names = []
        valid_cryptos = {}

        for name, df in all_data.items():
            if not df.empty and 'close' in df.columns:
                df['daily_return'] = df['close'].pct_change().dropna()
                if not df['daily_return'].empty:
                    returns_list.append(df['daily_return'])
                    crypto_names.append(name.replace('_USDT', ''))
                    valid_cryptos[name.replace('_USDT', '')] = df['daily_return']
                else:
                    logging.warning(f"Não há retornos diários válidos para {name}. Ignorando na ANOVA.")
            else:
                logging.warning(f"Dados inválidos ou sem coluna 'close' para {name}. Ignorando na ANOVA.")

        if len(returns_list) < 2:
            logging.warning("São necessárias pelo menos duas criptomoedas com retornos válidos para realizar ANOVA.")
            return

        f_statistic, p_value = stats.f_oneway(*returns_list)

        logging.info(f"  --- ANOVA entre Criptomoedas ---")
        logging.info(f"  F-Estatística: {f_statistic:.4f}")
        logging.info(f"  P-valor: {p_value:.4f}")
        logging.info(f"  Nível de Significância (alpha): {alpha}")

        anova_conclusion_all = ""
        if p_value < alpha:
            anova_conclusion_all = "Rejeitamos a hipótese nula. Há evidências significativas de que o retorno médio diário difere entre as criptomoedas analisadas."
            logging.info(f"  Conclusão: {anova_conclusion_all}")
            logging.info("  Realizando teste post hoc (Tukey HSD) para identificar diferenças...")

            # Preparar dados para Tukey HSD
            data_for_tukey = pd.DataFrame()
            for name, returns in valid_cryptos.items():
                temp_df = pd.DataFrame({'returns': returns, 'crypto': name})
                data_for_tukey = pd.concat([data_for_tukey, temp_df], ignore_index=True)

            if not data_for_tukey.empty:
                tukey_result = pairwise_tukeyhsd(endog=data_for_tukey['returns'], groups=data_for_tukey['crypto'], alpha=alpha)
                logging.info("\n" + str(tukey_result))

                # Plotar resultados do Tukey HSD
                plt.figure(figsize=(12, 8))
                tukey_result.plot_simultaneous()
                plt.title(f'Teste Post Hoc de Tukey HSD para Retornos Diários - {", ".join(crypto_names)}')
                plt.tight_layout()
                plot_path = os.path.join(save_folder, f"tukey_hsd_all_cryptos.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logging.info(f"Gráfico Tukey HSD salvo em: {plot_path}")
            else:
                logging.warning("Dados insuficientes para Tukey HSD após ANOVA.")

        else:
            anova_conclusion_all = "Não rejeitamos a hipótese nula. Não há evidências significativas de que o retorno médio diário difere entre as criptomoedas analisadas."
            logging.info(f"  Conclusão: {anova_conclusion_all}")

        # Salvar relatório ANOVA (todas as criptos)
        report_path_all = os.path.join(save_folder, "anova_report_all_cryptos.txt")
        with open(report_path_all, 'w') as f:
            f.write("Relatório de Análise de Variância (ANOVA) - Todas as Criptomoedas\n")
            f.write("-" * 70 + "\n")
            f.write(f"Criptomoedas Analisadas: {', '.join(crypto_names)}\n")
            f.write(f"Hipótese Nula (H0): Os retornos médios diários são iguais entre as criptomoedas.\n")
            f.write(f"Hipótese Alternativa (H1): Pelo menos um retorno médio diário difere.\n")
            f.write(f"Nível de Significância (alpha): {alpha}\n\n")
            f.write(f"F-Estatística: {f_statistic:.4f}\n")
            f.write(f"P-valor: {p_value:.4f}\n\n")
            f.write(f"Conclusão: {anova_conclusion_all}\n")
            if p_value < alpha:
                f.write("\nResultados do Teste Post Hoc (Tukey HSD):\n")
                if 'tukey_result' in locals():
                    f.write(str(tukey_result))
                else:
                    f.write("Não foi possível gerar resultados do Tukey HSD.\n")
        logging.info(f"Relatório ANOVA (todas as criptos) salvo em: {report_path_all}")


        # --- 11b. ANOVA por agrupamento (ex: por volatilidade média) ---
        logging.info("Realizando ANOVA para comparar retornos médios entre grupos de criptomoedas (por volatilidade média)...")

        # Calcular volatilidade média para agrupamento
        crypto_volatilities = {}
        for name, df in all_data.items():
            if not df.empty and 'close' in df.columns:
                df['daily_return'] = df['close'].pct_change().dropna()
                if not df['daily_return'].empty:
                    crypto_volatilities[name.replace('_USDT', '')] = df['daily_return'].std()
                else:
                    logging.warning(f"Não há retornos diários válidos para {name}. Ignorando no agrupamento por volatilidade.")

        if not crypto_volatilities or len(crypto_volatilities) < 2:
            logging.warning("Dados insuficientes para agrupar por volatilidade e realizar ANOVA.")
            return

        # Agrupar em "baixa", "média", "alta" volatilidade
        volatility_series = pd.Series(crypto_volatilities)
        low_threshold = volatility_series.quantile(0.33)
        high_threshold = volatility_series.quantile(0.66)

        groups = {
            'Baixa Volatilidade': [],
            'Média Volatilidade': [],
            'Alta Volatilidade': []
        }

        for crypto, vol in volatility_series.items():
            if vol <= low_threshold:
                groups['Baixa Volatilidade'].append(valid_cryptos[crypto])
            elif vol <= high_threshold:
                groups['Média Volatilidade'].append(valid_cryptos[crypto])
            else:
                groups['Alta Volatilidade'].append(valid_cryptos[crypto])

        # Remover grupos vazios para ANOVA
        groups_for_anova = {k: v for k, v in groups.items() if v}
        if len(groups_for_anova) < 2:
            logging.warning("Menos de 2 grupos de volatilidade formados. Não é possível realizar ANOVA por grupo.")
            return

        # Aplanar as listas de retornos para cada grupo
        group_returns_for_anova = [pd.concat(g) for g in groups_for_anova.values()]
        group_names_for_anova = list(groups_for_anova.keys())

        f_statistic_group, p_value_group = stats.f_oneway(*group_returns_for_anova)

        logging.info(f"  --- ANOVA entre Grupos de Volatilidade ---")
        logging.info(f"  Grupos Analisados: {group_names_for_anova}")
        logging.info(f"  F-Estatística: {f_statistic_group:.4f}")
        logging.info(f"  P-valor: {p_value_group:.4f}")
        logging.info(f"  Nível de Significância (alpha): {alpha}")

        anova_conclusion_group = ""
        if p_value_group < alpha:
            anova_conclusion_group = "Rejeitamos a hipótese nula. Há evidências significativas de que o retorno médio diário difere entre os grupos de volatilidade."
            logging.info(f"  Conclusão: {anova_conclusion_group}")
            logging.info("  Realizando teste post hoc (Tukey HSD) para identificar diferenças entre grupos...")

            # Preparar dados para Tukey HSD de grupos
            data_for_tukey_group = pd.DataFrame()
            for g_name, g_returns_list in groups_for_anova.items():
                for returns_series in g_returns_list:
                    temp_df = pd.DataFrame({'returns': returns_series, 'group': g_name})
                    data_for_tukey_group = pd.concat([data_for_tukey_group, temp_df], ignore_index=True)

            if not data_for_tukey_group.empty:
                tukey_result_group = pairwise_tukeyhsd(endog=data_for_tukey_group['returns'], groups=data_for_tukey_group['group'], alpha=alpha)
                logging.info("\n" + str(tukey_result_group))

                # Plotar resultados do Tukey HSD para grupos
                plt.figure(figsize=(12, 8))
                tukey_result_group.plot_simultaneous()
                plt.title(f'Teste Post Hoc de Tukey HSD para Retornos Diários - Grupos de Volatilidade')
                plt.tight_layout()
                plot_path_group = os.path.join(save_folder, f"tukey_hsd_volatility_groups.png")
                plt.savefig(plot_path_group, dpi=150)
                plt.close()
                logging.info(f"Gráfico Tukey HSD (grupos) salvo em: {plot_path_group}")
            else:
                logging.warning("Dados insuficientes para Tukey HSD para grupos após ANOVA.")

        else:
            anova_conclusion_group = "Não rejeitamos a hipótese nula. Não há evidências significativas de que o retorno médio diário difere entre os grupos de volatilidade."
            logging.info(f"  Conclusão: {anova_conclusion_group}")

        # Salvar relatório ANOVA (grupos)
        report_path_group = os.path.join(save_folder, "anova_report_volatility_groups.txt")
        with open(report_path_group, 'w') as f:
            f.write("Relatório de Análise de Variância (ANOVA) - Grupos de Volatilidade\n")
            f.write("-" * 70 + "\n")
            f.write(f"Grupos Analisados: {group_names_for_anova}\n")
            f.write(f"Hipótese Nula (H0): Os retornos médios diários são iguais entre os grupos de volatilidade.\n")
            f.write(f"Hipótese Alternativa (H1): Pelo menos um retorno médio diário difere entre os grupos.\n")
            f.write(f"Nível de Significância (alpha): {alpha}\n\n")
            f.write(f"F-Estatística: {f_statistic_group:.4f}\n")
            f.write(f"P-valor: {p_value_group:.4f}\n\n")
            f.write(f"Conclusão: {anova_conclusion_group}\n")
            if p_value_group < alpha:
                f.write("\nResultados do Teste Post Hoc (Tukey HSD):\n")
                if 'tukey_result_group' in locals():
                    f.write(str(tukey_result_group))
                else:
                    f.write("Não foi possível gerar resultados do Tukey HSD.\n")
        logging.info(f"Relatório ANOVA (grupos) salvo em: {report_path_group}")
    