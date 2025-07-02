# config.py
# Este arquivo pode ser usado para armazenar configurações globais do projeto.

# Exemplo: Lista de criptomoedas a serem baixadas
CRYPTOS_TO_DOWNLOAD = [
        "BTC", "ETH", "LTC", "XRP", "BCH",
        "XMR", "DASH", "ETC", "ZRX", "EOS"
    ]

QUOTE_CURRENCY = "USDT"
TIMEFRAME = "d" # Diário

# Pastas de saída
OUTPUT_FOLDER = "data/raw"
PLOTS_FOLDER = "figures/simple_plots"
ANALYSIS_FOLDER = "figures/analysis_plots"
PROCESSED_DATA_FOLDER = "data/processed"
MODELS_FOLDER = "models"
PROFIT_PLOTS_FOLDER = "figures/profit_plots"
STATS_REPORTS_FOLDER = "figures/statistical_reports"

# Parâmetros de modelo padrão
DEFAULT_KFOLDS = 5
DEFAULT_TARGET_RETURN_PERCENT = 0.01
DEFAULT_POLY_DEGREE = 2
    