CRIPTOS_PARA_BAIXAR = [
        "BTC", "ETH" #, "LTC", "XRP", "BCH",
        #"XMR", "DASH", "ETC", "ZRX", "EOS"
    ]

MOEDA_COTACAO = "USDT"
TIMEFRAME = "d" 

OUTPUT_FOLDER = 'output'
PROCESSED_DATA_FOLDER = 'processed'
MODELS_FOLDER = 'models'
PLOTS_FOLDER = 'plots'
ANALYSIS_FOLDER = 'analysis'
PROFIT_PLOTS_FOLDER = 'profit_plots'
STATS_REPORTS_FOLDER = 'stats_reports'

DEFAULT_KFOLDS = 5
DEFAULT_TARGET_RETURN_PERCENT = 0.01
DEFAULT_POLY_DEGREE = 2

MOVING_AVERAGE_WINDOWS = [7, 14, 30]
FEATURES_SELECIONADAS = [
    'high', 'low', 'sma_7', 'sma_14', 'sma_30',
    'close_lag5', 'macd', 'macd_signal', 'macd_diff',
    'bb_upper', 'bb_lower', 'bb_mavg', 'daily_return'
]

INITIAL_INVESTMENT = 1000.0

LOG_LEVEL = 'ERROR'

USE_USD_BRL = False

