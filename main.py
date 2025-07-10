# -*- coding: utf-8 -*-
"""
Ponto de Entrada Principal e Orquestrador do Pipeline.

Este script serve como o ponto de entrada principal para todo o projeto de
análise e modelagem de preços de criptoativos. Ele utiliza argumentos de
linha de comando para orquestrar e executar as diferentes etapas do
pipeline de forma modular.

O pipeline completo inclui as seguintes etapas, que podem ser executadas
individualmente com o argumento `--action` ou todas de uma vez (`--action all`):
1.  **Download (`download`):** Baixa e salva os dados históricos das criptomoedas.
2.  **Análise (`analyze`):** Gera estatísticas descritivas e gráficos de análise.
3.  **Engenharia de Features (`features`):** Cria novas features e indicadores técnicos.
4.  **Treinamento de Modelos (`train`):** Treina, avalia e compara modelos de ML.
5.  **Simulação de Lucro (`profit`):** Realiza backtesting de uma estratégia de
    investimento com base nas previsões dos modelos.
6.  **Testes Estatísticos (`stats`):** Executa testes de hipótese e ANOVA sobre os dados.

Exemplo de uso para treinar um modelo específico:
    $ python main.py --action train --crypto BTC --model RandomForest

Exemplo para executar todo o fluxo para uma única criptomoeda:
    $ python main.py --action all --crypto ETH
"""
import pandas as pd
import os
import logging
import argparse
import joblib
from typing import List, Dict, Any
from src.data_loader import load_crypto_data
from src.data_visualizer import plot_crypto_data
from src.data_analyzer import calculate_statistics, generate_analysis_plots, calculate_comparative_variability
from src.feature_engineering import create_moving_average_features, create_technical_features
from src.model_training import train_and_evaluate_model, compare_models, limpar_modelos_antigos
from src.prediction_profit import simulate_investment_and_profit
from src.statistical_tests import perform_hypothesis_test, perform_anova_analysis
from src.feature_engineering import enrich_with_external_features
from src.utils import (
    get_pair_key,
    get_raw_data_filepath,
    get_processed_data_filepath
)
from src.model_training import get_best_model_by_mse

from config import (
    CRIPTOS_PARA_BAIXAR,
    MOEDA_COTACAO,
    TIMEFRAME,
    OUTPUT_FOLDER,
    PROCESSED_DATA_FOLDER,
    MODELS_FOLDER,
    PLOTS_FOLDER,
    ANALYSIS_FOLDER,
    PROFIT_PLOTS_FOLDER,
    STATS_REPORTS_FOLDER,
    DEFAULT_KFOLDS,
    DEFAULT_TARGET_RETURN_PERCENT,
    DEFAULT_POLY_DEGREE,
    LOG_LEVEL,
    MOVING_AVERAGE_WINDOWS,
    FEATURES_SELECIONADAS,
    INITIAL_INVESTMENT,
    USE_USD_BRL,
    N_ESTIMATORS_RF
)

def setup_logging(level_str: str = 'ERROR'):
    """
    Configura o sistema de logging global para a aplicação.

    Args:
        level_str (str, optional): O nível de logging desejado, em formato de string
                                   (ex: 'INFO', 'DEBUG', 'WARNING', 'ERROR').
                                   O padrão é 'ERROR'.
    """
    level = getattr(logging, level_str.upper(), logging.ERROR)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )

def main():
    """
    Executa o pipeline principal do projeto de análise e previsão de preços de criptomoedas.

    Este pipeline inclui:
    - Download e enriquecimento de dados de criptomoedas.
    - Análise estatística e geração de gráficos.
    - Engenharia de features (médias móveis, indicadores técnicos, etc).
    - Treinamento e avaliação de modelos de machine learning.
    - Simulação de estratégias de investimento.
    - Testes estatísticos sobre os resultados.

    O comportamento é controlado pelos argumentos de linha de comando, permitindo executar etapas específicas ou todo o fluxo.
    """
    setup_logging(LOG_LEVEL)

    parser = argparse.ArgumentParser(description="Análise e Previsão de Preços de Criptomoedas.")
    parser.add_argument('--action', type=str, default='all',
                        choices=['all', 'download', 'analyze', 'features', 'train', 'profit', 'stats'],
                        help="Ação a ser executada.")
    parser.add_argument('--crypto', type=str, default='all',
                        help="Símbolo da criptomoeda para processar (ex: BTC). 'all' para todas.")
    parser.add_argument('--model', type=str, default=None,
                        choices=['MLP', 'Linear', 'Polynomial', 'RandomForest'],
                        help="Modelo a ser usado para treinamento.")
    parser.add_argument('--kfolds', type=int, default=DEFAULT_KFOLDS,
                        help="Número de folds para K-fold cross-validation.")
    parser.add_argument('--target_return_percent', type=float, default=DEFAULT_TARGET_RETURN_PERCENT,
                        help="Retorno esperado médio (%) para o teste de hipótese.")
    parser.add_argument('--poly_degree', type=int, default=DEFAULT_POLY_DEGREE,
                        help="Grau máximo para a regressão polinomial.")
    parser.add_argument('--validation_split', type=float, default=0.3,
                    help="Fração dos dados usada como hold-out para validação final (ex: 0.3 = 30%, 0.0 = sem separação pra hold-out).")
    parser.add_argument('--n_estimators', type=int, default=N_ESTIMATORS_RF,
                        help="Número de estimadores para o modelo RandomForest.")
    parser.add_argument('--force_download', action='store_true',
                        help="Força o download dos dados mesmo que o arquivo já exista.")
    parser.add_argument("--use_usd_brl", action="store_true", default=USE_USD_BRL,
                        help="Incluir cotação USD/BRL como feature externa.")
    args = parser.parse_args()

    for folder in [OUTPUT_FOLDER, PLOTS_FOLDER, ANALYSIS_FOLDER, PROCESSED_DATA_FOLDER,
                   MODELS_FOLDER, PROFIT_PLOTS_FOLDER, STATS_REPORTS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    all_dfs: Dict[str, pd.DataFrame] = {}
    all_processed_dfs: Dict[str, pd.DataFrame] = {}

    cryptos_to_process = CRIPTOS_PARA_BAIXAR if args.crypto == 'all' else [args.crypto.upper()]

    if args.action in ['all', 'download', 'analyze', 'features', 'stats']:
        logging.info("Carregando dados brutos...")
        for simbolo_base in cryptos_to_process:
            pair_key = get_pair_key(simbolo_base)
            caminho_arquivo = get_raw_data_filepath(simbolo_base)
            
            if args.action in ['all', 'download']:
                logging.info(f"Processando {simbolo_base}...")
                df = load_crypto_data(base_symbol=simbolo_base, quote_symbol=MOEDA_COTACAO, timeframe=TIMEFRAME, calculate_indicators=False)                

                if df is not None and not df.empty:
                    if args.use_usd_brl:
                        df = enrich_with_external_features(df, use_usd_brl=True)
                    df.to_csv(caminho_arquivo, index=False)
                    all_dfs[pair_key] = df
                    logging.info(f"Dados brutos para {simbolo_base} processados e salvos.")
            elif os.path.exists(caminho_arquivo):
                try:
                    all_dfs[pair_key] = pd.read_csv(caminho_arquivo)
                except Exception as e:
                    logging.error(f"Falha ao ler o arquivo {caminho_arquivo}: {e}")
            else:
                logging.warning(f"Arquivo de dados brutos não encontrado para {pair_key}. Execute a ação 'download'.")

    if args.action in ['all', 'train', 'profit']:
        logging.info("Carregando dados processados com features...")
        for simbolo_base in cryptos_to_process:
            pair_key = get_pair_key(simbolo_base)
            caminho_arquivo = get_processed_data_filepath(simbolo_base)
            if os.path.exists(caminho_arquivo):
                try:
                    all_processed_dfs[pair_key] = pd.read_csv(caminho_arquivo)
                except Exception as e:
                    logging.error(f"Falha ao ler o arquivo processado {caminho_arquivo}: {e}")
            else:
                logging.warning(f"Arquivo de dados processados não encontrado para {pair_key}. Execute a ação 'features'.")

    if args.action in ['all', 'analyze']:
        if not all_dfs:
            logging.error("Nenhum dado bruto disponível para a ação 'analyze'.")
        else:
            logging.info("Iniciando análises estatísticas e geração de gráficos.")
            for pair_key, df in all_dfs.items():
                stats = calculate_statistics(df)
                generate_analysis_plots(df, pair_name=pair_key, save_folder=ANALYSIS_FOLDER)
                plot_crypto_data(df, pair_name=pair_key, save_folder=PLOTS_FOLDER)
            variability_df = calculate_comparative_variability(all_dfs)
            logging.info(f"\n*** Análise Comparativa de Variabilidade ***\n{variability_df.to_string()}")


    if args.action in ['all', 'features']:
        if not all_dfs:
            logging.error("Nenhum dado bruto disponível para a ação 'features'.")
        else:
            logging.info("Iniciando engenharia de features.")
            for pair_key, df in all_dfs.items():
                logging.info(f"Criando features para {pair_key}...")
                df_featured = create_technical_features(df.copy())

                all_processed_dfs[pair_key] = df_featured

                simbolo_base = pair_key.split('_')[0]
                processed_filepath = get_processed_data_filepath(simbolo_base)
                df_featured.to_csv(processed_filepath, index=False)
                logging.info(f"Features para {pair_key} salvas em: {processed_filepath}")

    if args.action in ['all', 'train']:
        if not all_processed_dfs:
            logging.error("Nenhum dado processado disponível para a ação 'train'.")

        else:
            logging.info("Iniciando treinamento e avaliação de modelos.")
            for pair_key, df_featured in all_processed_dfs.items():
                logging.info(f"Processando modelos para {pair_key}...")
                features = [col for col in FEATURES_SELECIONADAS if col in df_featured.columns]
                X = df_featured[features]
                y = df_featured['close']
                mask = ~(X.isna().any(axis=1) | y.isna())
                X_clean, y_clean = X[mask], y[mask]

                if X_clean.empty:
                    logging.warning(f"Sem dados suficientes para treinar modelos para {pair_key} após limpeza.")
                    continue

                
                # Limpa modelos antigos antes de iniciar o treino, garantindo que apenas
                # o modelo mais recente (especificado ou selecionado automaticamente)
                # será considerado na fase de simulação de lucro.
                limpar_modelos_antigos(pair_key, MODELS_FOLDER) 

                #Inicia o treinamento de modelos
                if args.model:
                    #usa o modelo especificado pelo usuário
                    train_and_evaluate_model(X_clean, y_clean, model_type=args.model, kfolds=args.kfolds,
                                            pair_name=pair_key, models_folder=MODELS_FOLDER,
                                            poly_degree=args.poly_degree, n_estimators=args.n_estimators,
                                            test_size=args.validation_split)
                    compare_models(X_clean, y_clean, kfolds=args.kfolds, pair_name=pair_key,
                                plots_folder=ANALYSIS_FOLDER, poly_degree=args.poly_degree,
                                n_estimators=args.n_estimators, test_size=args.validation_split)
                else:
                    # modo automático: encontra e salva o melhor modelo
                    best_model, best_name = get_best_model_by_mse(
                        X_clean, y_clean,
                        kfolds=args.kfolds,
                        poly_degree=args.poly_degree,
                        n_estimators=args.n_estimators
                    )

                    if best_model is not None:
                        best_model.fit(X_clean, y_clean)
                        model_path = os.path.join(MODELS_FOLDER, f"{best_name.lower()}_{pair_key}.pkl")
                        joblib.dump(best_model, model_path)
                        logging.info(f"Melhor modelo ({best_name}) salvo em: {model_path}")
                        compare_models(X_clean, y_clean, kfolds=args.kfolds, pair_name=pair_key,
                                    plots_folder=ANALYSIS_FOLDER, poly_degree=args.poly_degree,
                                    n_estimators=args.n_estimators, test_size=args.validation_split)
                    else:
                        logging.warning(f"Não foi possível determinar o melhor modelo para {pair_key}.")

               

    if args.action in ['all', 'profit']:
        if not all_processed_dfs:
            logging.error("Nenhum dado processado disponível para a ação 'profit'.")
        else:
            logging.info("Iniciando simulação de lucro.")
            for pair_key, df_featured in all_processed_dfs.items():
                logging.info(f"Simulando lucro para {pair_key}...")
                features = [col for col in FEATURES_SELECIONADAS if col in df_featured.columns]
                X = df_featured[features]
                y = df_featured['close']
                dates = pd.to_datetime(df_featured['date'])
                simulate_investment_and_profit(X, y, dates, pair_name=pair_key,
                                               models_folder=MODELS_FOLDER,
                                               profit_plots_folder=PROFIT_PLOTS_FOLDER,
                                               initial_investment=INITIAL_INVESTMENT)
    if args.action in ['all', 'stats']:
        if not all_dfs:
            logging.error("Nenhum dado bruto disponível para a ação 'stats'.")
        else:
            logging.info("Iniciando testes estatísticos avançados.")
            for pair_key, df in all_dfs.items():
                perform_hypothesis_test(df, pair_key, args.target_return_percent, STATS_REPORTS_FOLDER)
            
            perform_anova_analysis(all_dfs, STATS_REPORTS_FOLDER)

    logging.info("FLUXO DE TRABALHO CONCLUÍDO!")


if __name__ == '__main__':
    main()