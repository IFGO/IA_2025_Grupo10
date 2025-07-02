# main.py
import argparse
import logging
from src.data_loader import load_crypto_data
# Importar outras funções necessárias

def main(args):
    logging.info(f"Iniciando processo para a criptomoeda: {args.crypto}")

    # Exemplo de fluxo
    # 1. Carregar os dados
    # raw_data = load_crypto_data(args.crypto)

    # 2. Realizar análises estatísticas
    # run_statistical_analysis(raw_data)

    # 3. Engenharia de features
    # features = create_features(raw_data)

    # 4. Treinamento e avaliação do modelo
    # train_and_evaluate(features, args.model, args.kfolds)

    logging.info("Processo finalizado com sucesso.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Projeto de Previsão de Preços de Criptomoedas.")
    parser.add_argument('--crypto', type=str, default='BTC', help='Símbolo da criptomoeda (ex: BTC, ETH).')
    parser.add_argument('--model', type=str, default='MLP', choices=['MLP', 'Linear', 'Polynomial'], help='Modelo a ser treinado.')
    parser.add_argument('--kfolds', type=int, default=5, help='Número de folds para a validação cruzada.')

    # Adicionar outros argumentos conforme necessário (ex: --task, --start_date, etc.)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    main(args)