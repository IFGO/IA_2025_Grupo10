# main_workflow.py (Versão Corrigida e Completa)
import pandas as pd
import os
import logging
from typing import List, Dict, Any

# --- Importações dos Módulos Personalizados ---
from data_loader import load_crypto_data
from data_visualizer import plot_crypto_data
# CORRIGIDO: Importa as funções corretas do nosso analyzer revisado
from data_analyzer import calculate_statistics, generate_analysis_plots, calculate_comparative_variability

def main():
    """
    Função principal que orquestra o download, o relatório, a visualização e a análise dos dados.
    """
    # --- 1. Configuração ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    criptos_para_baixar = [
        "BTC", "ETH", "LTC", "XRP", "BCH", 
        "XMR", "DASH","ETC", "ZRX", "EOS" # 10 moedas
    ]
    moeda_cotacao = "USDT"
    timeframe = "d"
    
    output_folder = "crypto_datasets"
    plots_folder = "crypto_plots"
    analysis_folder = "crypto_analysis" # Pasta para relatórios e gráficos de análise

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(analysis_folder, exist_ok=True)

    summary_logs: List[Dict[str, Any]] = []
    all_stats: Dict[str, pd.Series] = {} # Dicionário para estatísticas
    all_dfs: Dict[str, pd.DataFrame] = {} # ADICIONADO: Dicionário para guardar os dataframes para análise comparativa

    # --- 2. Download, Análise Individual e Salvamento ---
    logging.info(f"Iniciando processo de download e análise para {len(criptos_para_baixar)} criptomoedas.")
    print("-" * 70)
    
    for simbolo_base in criptos_para_baixar:
        df = load_crypto_data(base_symbol=simbolo_base, quote_symbol=moeda_cotacao, timeframe=timeframe)

        log_info = {
            "Par de Moedas": f"{simbolo_base.upper()}/{moeda_cotacao.upper()}",
            "Status": "Falha", "Total de Registros": 0, "Data de Início": "N/A",
            "Data de Fim": "N/A", "Preço Médio (Close)": 0.0, "Arquivo Salvo": "Nenhum"
        }

        if df is not None and not df.empty:
            nome_arquivo = f"{simbolo_base.upper()}_{moeda_cotacao.upper()}_{timeframe}.csv"
            caminho_arquivo = os.path.join(output_folder, nome_arquivo)
            df.to_csv(caminho_arquivo, index=False)
            
            log_info.update({
                "Status": "Sucesso", "Total de Registros": len(df),
                "Data de Início": df['date'].min().strftime('%Y-%m-%d'),
                "Data de Fim": df['date'].max().strftime('%Y-%m-%d'),
                "Preço Médio (Close)": round(df['close'].mean(), 4),
                "Arquivo Salvo": nome_arquivo
            })
            logging.info(f"Sucesso ao processar e salvar {simbolo_base}.")
            
            # --- Seção de Análise Individual ---
            pair_key = f"{simbolo_base.upper()}_{moeda_cotacao.upper()}_{timeframe.upper()}"
            all_dfs[pair_key] = df # Guarda o dataframe para análise posterior

            stats = calculate_statistics(df)
            all_stats[log_info["Par de Moedas"]] = stats
            
            generate_analysis_plots(df, pair_name=pair_key, save_folder=analysis_folder)
        
        summary_logs.append(log_info)

    # --- 3. Relatório de Download ---
    print("-" * 70)
    logging.info("Gerando relatório de resumo do download...")
    df_summary = pd.DataFrame(summary_logs)
    df_summary.to_csv(os.path.join(output_folder, "relatorio_download.csv"), index=False)
    print("\n*** Relatório Final do Download ***\n")
    print(df_summary.to_string())

    # --- 4. Relatório de Análise Estatística ---
    print("-" * 70)
    logging.info("Gerando relatório de análise estatística...")
    if all_stats:
        df_stats = pd.DataFrame.from_dict(all_stats, orient='index')
        df_stats.to_csv(os.path.join(analysis_folder, "relatorio_estatistico_descritivo.csv"))
        print("\n*** Relatório de Análise Estatística (Preço de Fechamento) ***\n")
        print(df_stats)
        print(f"\nGráficos de análise consolidados salvos na pasta: '{analysis_folder}'")
    else:
        logging.warning("Nenhuma estatística foi calculada pois nenhum download teve sucesso.")

    # --- 5. ADICIONADO: Relatório de Análise Comparativa ---
    print("-" * 70)
    logging.info("Gerando relatório de análise de variabilidade comparativa...")
    if all_dfs:
        df_variability = calculate_comparative_variability(all_dfs)
        df_variability.to_csv(os.path.join(analysis_folder, "relatorio_comparativo_volatilidade.csv"), index=False)
        print("\n*** Análise Comparativa de Volatilidade (Ordenado pela Maior Relativa) ***\n")
        print(df_variability.to_string())
    else:
        logging.warning("Nenhuma análise comparativa pôde ser feita.")
        
    # --- 6. Visualização (Séries Temporais Simples) ---
    print("-" * 70)
    logging.info("Gerando visualizações simples das séries temporais...")
    for log in summary_logs:
        if log["Status"] == "Sucesso":
            csv_path = os.path.join(output_folder, log["Arquivo Salvo"])
            plot_crypto_data(csv_path=csv_path, save_folder=plots_folder)
            
    logging.info(f"Processo de visualização finalizado. Verifique a pasta '{plots_folder}'.")
    print("\nFLUXO DE TRABALHO CONCLUÍDO!")

if __name__ == '__main__':
    main()