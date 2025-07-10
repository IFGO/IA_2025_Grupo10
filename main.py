import pandas as pd
import os
import logging
import argparse
from typing import List, Dict, Any

from src.data_loader import load_crypto_data
from src.data_visualizer import plot_crypto_data
from src.data_analyzer import calculate_statistics, generate_analysis_plots, calculate_comparative_variability
from src.feature_engineering import create_moving_average_features, create_technical_features
from src.model_training import train_and_evaluate_model, compare_models
from src.prediction_profit import simulate_investment_and_profit
from src.statistical_tests import perform_hypothesis_test, perform_anova_analysis
from src.feature_engineering import enrich_with_external_features

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
    parser.add_argument('--model', type=str, default='MLP',
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

    summary_logs: List[Dict[str, Any]] = []
    all_stats: Dict[str, pd.Series] = {}
    all_dfs: Dict[str, pd.DataFrame] = {}
    all_processed_dfs: Dict[str, pd.DataFrame] = {}

    if args.action in ['all', 'download']:
        if args.crypto == 'all' or args.crypto == '':
            logging.info(f"Iniciando processo de download para {len(CRIPTOS_PARA_BAIXAR)} criptomoedas.")
        else:
            logging.info(f"Iniciando processo de download para a criptomoeda: {args.crypto.upper()}.")

        for simbolo_base in CRIPTOS_PARA_BAIXAR:
            if args.crypto != 'all' and simbolo_base != args.crypto:
                continue
            
            nome_arquivo = f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}_{TIMEFRAME}.csv"
            caminho_arquivo = os.path.join(OUTPUT_FOLDER, nome_arquivo)
            
            # Verifica se o arquivo já existe e se não deve forçar o download, o and not use_usd_brl garantirá que toda vez que use_usd_brl for True, o download será feito para realizar o enriquecimento com a cotação USD/BRL
            if not args.force_download and os.path.exists(caminho_arquivo) and not args.use_usd_brl:
                df = pd.read_csv(caminho_arquivo)
                all_dfs[f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}"] = df
                continue

            df = load_crypto_data(base_symbol=simbolo_base, quote_symbol=MOEDA_COTACAO, timeframe=TIMEFRAME)

            if df is not None and not df.empty:
                if args.use_usd_brl:
                    df = enrich_with_external_features(df, use_usd_brl=True)
                df.to_csv(caminho_arquivo, index=False)
                all_dfs[f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}"] = df
                logging.info(f"Sucesso ao processar e salvar {simbolo_base}.")

    if args.action in ['all', 'analyze']:
        logging.info("Iniciando análises estatísticas e geração de gráficos.")

       
        #caso esteja executando só analyze, sem ter executado all
        if args.action == 'analyze':
            for simbolo_base in CRIPTOS_PARA_BAIXAR:
                if args.crypto != 'all' and simbolo_base != args.crypto:
                    continue
                
                nome_arquivo = f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}_{TIMEFRAME}.csv"
                caminho_arquivo = os.path.join(OUTPUT_FOLDER, nome_arquivo)
                
                # Verifica se o arquivo já existe, se nao conseguir ler o arquivo com pd.read_csv(caminho_arquivo) ou o arquivo estiver vazio logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto. Você deve executar antes python main.py --action download.")
                if os.path.exists(caminho_arquivo):
                    df = pd.read_csv(caminho_arquivo)
                    all_dfs[f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}"] = df
                    continue
                #se o arquivo não existir ou estiver vazio, apresenta a mensagem
                logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto.")
                
        
        #se all_dfs estiver vazio, significa que não foram baixados dados, então não faz sentido continuar
        if not all_dfs:
            logging.error("Nenhum dado disponível para análise. Execute primeiro a ação 'download' com python main.py --action download, para criar os arquivos de todas as moedas ou informando a criptomoeda desejada com o parâmetro crypto.")
            return

        for pair_key, df in all_dfs.items():
            stats = calculate_statistics(df)
            all_stats[pair_key] = stats
            generate_analysis_plots(df, pair_name=pair_key, save_folder=ANALYSIS_FOLDER)
            plot_crypto_data(df, pair_name=pair_key, save_folder=PLOTS_FOLDER)

    if args.action in ['all', 'features']:
        logging.info("Iniciando engenharia de features.")

        #caso esteja executando só features, sem ter executado all
        if args.action == 'features':
            for simbolo_base in CRIPTOS_PARA_BAIXAR:
                if args.crypto != 'all' and simbolo_base != args.crypto:
                    continue
                
                nome_arquivo = f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}_{TIMEFRAME}.csv"
                caminho_arquivo = os.path.join(OUTPUT_FOLDER, nome_arquivo)
                
                # Verifica se o arquivo já existe, se nao conseguir ler o arquivo com pd.read_csv(caminho_arquivo) ou o arquivo estiver vazio logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto. Você deve executar antes python main.py --action download.")
                if os.path.exists(caminho_arquivo):
                    df = pd.read_csv(caminho_arquivo)
                    all_dfs[f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}"] = df
                    continue
                #se o arquivo não existir ou estiver vazio, apresenta a mensagem
                logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto. .")
                
        
        #se all_dfs estiver vazio, significa que não foram baixados dados, então não faz sentido continuar
        if not all_dfs:
            logging.error("Nenhum dado disponível para análise. Execute primeiro a ação 'download' com python main.py --action download, para criar os arquivos de todas as moedas ou informando a criptomoeda desejada com o parâmetro crypto.")
            return


        for pair_key, df in all_dfs.items():
            if args.crypto != 'all' and pair_key.split('_')[0] != args.crypto:
                continue
            logging.info(f"Criando features para {pair_key}...")
            df_featured = create_moving_average_features(df.copy(), MOVING_AVERAGE_WINDOWS)
            df_featured = create_technical_features(df_featured.copy())
            processed_filename = f"featured_{pair_key}.csv"
            processed_filepath = os.path.join(PROCESSED_DATA_FOLDER, processed_filename)
            df_featured.to_csv(processed_filepath, index=False)
            all_processed_dfs[pair_key] = df_featured
            logging.info(f"Features para {pair_key} salvas em: {processed_filepath}")

    if args.action in ['all', 'train']:
        logging.info("Iniciando treinamento e avaliação de modelos.")
        #==============================================
        
        #caso esteja executando só train, sem ter executado all
        if args.action == 'train':
            for simbolo_base in CRIPTOS_PARA_BAIXAR:
                if args.crypto != 'all' and simbolo_base != args.crypto:
                    continue
                
                nome_arquivo = f"featured_{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}.csv"
                caminho_arquivo = os.path.join(PROCESSED_DATA_FOLDER, nome_arquivo)
                
                # Verifica se o arquivo já existe, se nao conseguir ler o arquivo com pd.read_csv(caminho_arquivo) ou o arquivo estiver vazio logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto. Você deve executar antes python main.py --action features (que tem como pré-requisito main.py --action download).")
                if os.path.exists(caminho_arquivo):
                    df = pd.read_csv(caminho_arquivo)
                    all_processed_dfs[f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}"] = df
                    continue
                #se o arquivo não existir ou estiver vazio, apresenta a mensagem
                logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto.")
                
        
        #se all_processed_dfs estiver vazio, significa que não foram processados os arquivos csv com os dados, então não faz sentido continuar
        if not all_processed_dfs:
            logging.error("Nenhum dado disponível para análise. Execute primeiro a ação 'features' com python main.py --action features, para criar os arquivos de todas as moedas ou informando a criptomoeda desejada com o parâmetro crypto. Lembrando, para esta ação a --action download deve ser executada antes.")
            return
        
        
        
        
        #==============================
        for pair_key, df_featured in all_processed_dfs.items():
            if args.crypto != 'all' and pair_key.split('_')[0] != args.crypto:
                continue
            logging.info(f"Treinando e avaliando modelos para {pair_key}...")
            features = [col for col in FEATURES_SELECIONADAS if col in df_featured.columns]
            X = df_featured[features]
            y = df_featured['close']
            # Remove linhas com valores ausentes
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[mask]
            y_clean = y[mask]
            train_and_evaluate_model(X_clean, y_clean, model_type=args.model, kfolds=args.kfolds,
                                     pair_name=pair_key, models_folder=MODELS_FOLDER,
                                     poly_degree=args.poly_degree,
                                     n_estimators=args.n_estimators,
                                     test_size=args.validation_split)
            compare_models(X_clean, y_clean, kfolds=args.kfolds, pair_name=pair_key,
                           plots_folder=ANALYSIS_FOLDER, poly_degree=args.poly_degree,
                           n_estimators=args.n_estimators,
                           test_size=args.validation_split)

    if args.action in ['all', 'profit']:
        logging.info("Iniciando simulação de lucro.")
        #==============================================
        
        #caso esteja executando só profit, sem ter executado all
        if args.action == 'profit':
            for simbolo_base in CRIPTOS_PARA_BAIXAR:
                if args.crypto != 'all' and simbolo_base != args.crypto:
                    continue
                
                nome_arquivo = f"featured_{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}.csv"
                caminho_arquivo = os.path.join(PROCESSED_DATA_FOLDER, nome_arquivo)
                
                # Verifica se o arquivo já existe, se nao conseguir ler o arquivo com pd.read_csv(caminho_arquivo) ou o arquivo estiver vazio logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto. Você deve executar antes python main.py --action features (que tem como pré-requisito main.py --action download).")
                if os.path.exists(caminho_arquivo):
                    df = pd.read_csv(caminho_arquivo)
                    all_processed_dfs[f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}"] = df
                    continue
                #se o arquivo não existir ou estiver vazio, apresenta a mensagem
                logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto.")                
        
        #se all_processed_dfs estiver vazio, significa que não foram processados os arquivos csv com os dados, então não faz sentido continuar
        if not all_processed_dfs:
            logging.error("Nenhum dado disponível para análise. Execute primeiro a ação 'features' com python main.py --action features, para criar os arquivos de todas as moedas ou informando a criptomoeda desejada com o parâmetro crypto. Lembrando, para esta ação a --action download deve ser executada antes.")
            return
        
        
        
        
        #==============================
        for pair_key, df_featured in all_processed_dfs.items():
            if args.crypto != 'all' and pair_key.split('_')[0] != args.crypto:
                continue
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
        logging.info("Iniciando testes estatísticos avançados.")

        #caso esteja executando só stats, sem ter executado all
        if args.action == 'stats':
            for simbolo_base in CRIPTOS_PARA_BAIXAR:
                if args.crypto != 'all' and simbolo_base != args.crypto:
                    continue
                
                nome_arquivo = f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}_{TIMEFRAME}.csv"
                caminho_arquivo = os.path.join(OUTPUT_FOLDER, nome_arquivo)
                
                # Verifica se o arquivo já existe, se nao conseguir ler o arquivo com pd.read_csv(caminho_arquivo) ou o arquivo estiver vazio logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto. Você deve executar antes python main.py --action download.")
                if os.path.exists(caminho_arquivo):
                    df = pd.read_csv(caminho_arquivo)
                    all_dfs[f"{simbolo_base.upper()}_{MOEDA_COTACAO.upper()}"] = df
                    continue
                #se o arquivo não existir ou estiver vazio, apresenta a mensagem
                logging.error(f"Erro ao ler o arquivo {caminho_arquivo}. Verifique se o arquivo existe e está no formato correto.")
                
        
        #se all_dfs estiver vazio, significa que não foram baixados dados, então não faz sentido continuar
        if not all_dfs:
            logging.error("Nenhum dado disponível para análise. Execute primeiro a ação 'download' com python main.py --action download, para criar os arquivos de todas as moedas ou informando a criptomoeda desejada com o parâmetro crypto.")
            return


        for pair_key, df in all_dfs.items():
            if args.crypto != 'all' and pair_key.split('_')[0] != args.crypto:
                continue
            perform_hypothesis_test(df, pair_key, args.target_return_percent, STATS_REPORTS_FOLDER)
        perform_anova_analysis(all_processed_dfs, STATS_REPORTS_FOLDER)

    logging.info("FLUXO DE TRABALHO CONCLUÍDO!")

if __name__ == '__main__':
    main()