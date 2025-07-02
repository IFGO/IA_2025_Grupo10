import argparse
import sys
import logging
import os
from data_load import load_crypto_data
from statistics_module import run_statistics
from models import train_model, simulate_investment

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def clean_figure_files(symbol: str, prefix: str):
    folder = "figures"
    if not os.path.exists(folder):
        return
    for file in os.listdir(folder):
        if file.startswith(prefix) and file.endswith(f"_{symbol}.png"):
            try:
                os.remove(os.path.join(folder, file))
                logging.info(f"Removido: {file}")
            except Exception as e:
                logging.warning(f"Erro ao remover {file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analisador e Previsor de Criptomoedas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: stats
    stats_parser = subparsers.add_parser("stats", help="Executa estatísticas exploratórias")
    stats_parser.add_argument("--cryptos", nargs="+", required=True, help="Lista de criptomoedas a analisar")

    # Subcomando: train
    train_parser = subparsers.add_parser("train", help="Treina modelo para previsão de preço")
    train_parser.add_argument("--crypto", required=True, help="Nome da criptomoeda")
    train_parser.add_argument("--model", required=True, choices=["mlp", "linear"], help="Modelo a utilizar")

    # Subcomando: simulate
    simulate_parser = subparsers.add_parser("simulate", help="Simula investimento baseado em previsão")
    simulate_parser.add_argument("--crypto", required=True, help="Criptomoeda a utilizar")
    simulate_parser.add_argument("--model", required=True, choices=["mlp", "linear"], help="Modelo preditivo")
    simulate_parser.add_argument("--initial-investment", type=float, default=1000.0, help="Capital inicial em USD")

    args = parser.parse_args()

    if args.command == "stats":
        for crypto in args.cryptos:
            try:
                clean_figure_files(crypto, "boxplot")
                clean_figure_files(crypto, "fechamento_estatisticas")
                df = load_crypto_data(crypto)
                run_statistics(df, crypto)
            except FileNotFoundError:
                logging.warning(f"Arquivo para {crypto} não encontrado. Pulando.")
            except Exception as e:
                logging.error(f"Erro inesperado ao processar {crypto}: {e}")

    elif args.command == "train":
        clean_figure_files(args.crypto, f"dispersao_{args.model}")
        try:
            df = load_crypto_data(args.crypto)
            train_model(df, args.crypto, args.model)
        except Exception as e:
            logging.error(f"Erro no treinamento: {e}")

    elif args.command == "simulate":
        clean_figure_files(args.crypto, f"capital_{args.model}")
        try:
            df = load_crypto_data(args.crypto)
            simulate_investment(df, args.crypto, args.model, args.initial_investment)
        except Exception as e:
            logging.error(f"Erro na simulação: {e}")

if __name__ == "__main__":
    main()
