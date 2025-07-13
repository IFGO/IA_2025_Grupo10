# -*- coding: utf-8 -*-
"""
Arquivo de Configuração Central do Projeto.

Este módulo centraliza todas as variáveis de configuração e parâmetros
utilizados nas diferentes etapas do projeto de análise e modelagem de
criptoativos. O objetivo é facilitar a manutenção, a experimentação e a
reprodutibilidade, permitindo que as configurações sejam alteradas em um
único local sem a necessidade de modificar o código-fonte dos scripts
principais.

As configurações estão agrupadas nas seguintes categorias:

- **Parâmetros de Aquisição de Dados:**
  - `CRIPTOS_PARA_BAIXAR`: Lista dos símbolos das criptomoedas a serem analisadas.
  - `MOEDA_COTACAO`: A moeda base para a cotação (ex: 'USDT').
  - `TIMEFRAME`: O intervalo de tempo dos dados (ex: 'd' para diário).

- **Diretórios e Pastas:**
  - Define os caminhos para salvar dados brutos, processados, modelos
    treinados, gráficos, relatórios estatísticos e plots de lucro.

- **Parâmetros de Modelagem:**
  - `DEFAULT_KFOLDS`: Número padrão de folds para a validação cruzada.
  - `DEFAULT_TARGET_RETURN_PERCENT`: Alvo de retorno para testes de hipótese.
  - `DEFAULT_POLY_DEGREE`: Grau padrão para regressão polinomial.
  - `N_ESTIMATORS_RF`: Número de estimadores para o modelo RandomForest.

- **Parâmetros de Engenharia de Features:**
  - `MOVING_AVERAGE_WINDOWS`: Janelas para cálculo de médias móveis.
  - `FEATURES_CANDIDATAS`: Lista de features a serem usadas no treinamento do modelo inicialmente. Ao final, o software escolhe as melhores.
  - `USE_USD_BRL`: Flag para controlar a adição de dados externos (cotação USD/BRL).

- **Parâmetros de Simulação e Log:**
  - `INITIAL_INVESTMENT`: Valor do investimento inicial para simulações de lucro.
  - `LOG_LEVEL`: Nível de verbosidade dos logs (ex: 'INFO', 'DEBUG').
"""

CRIPTOS_PARA_BAIXAR = [
    "BTC",
    "ETH",
    # "LTC",
    # "XRP",
    # "BCH",
    # "XMR",
    "DASH",
    # "ETC",
    # "ZRX",
    # "EOS",
]

MOEDA_COTACAO = "USDT"
TIMEFRAME = "d"

RAW_FILENAME_TEMPLATE = "{base}_{quote}_{timeframe}.csv"
FEATURED_FILENAME_TEMPLATE = "featured_{base}_{quote}.csv"
MODEL_FILENAME_TEMPLATE = "{model_type}_{base}_{quote}.pkl"

OUTPUT_FOLDER = "data/output"  # Local de saída dos dados pre-processados dos arquivos baixados: remove nans, converte tipos, ordena por data, refaz os índices, etc.
PROCESSED_DATA_FOLDER = "data/processed"
MODELS_FOLDER = "data/models"
PLOTS_FOLDER = "grafico/plots"
ANALYSIS_FOLDER = "grafico/analysis"
PROFIT_PLOTS_FOLDER = "grafico/profit_plots"
STATS_REPORTS_FOLDER = "data/stats_reports"

DEFAULT_KFOLDS = 5
DEFAULT_TARGET_RETURN_PERCENT = 0.01
DEFAULT_POLY_DEGREE = 2
N_ESTIMATORS_RF = 100
DEFAULT_K_BEST = 6  # Número de features a serem selecionadas pelo SelectKBest fora a cotação do dólar, caso acionado por --use_usd_brl
DEFAULT_VALIDATION_SPLIT = (
    0.2  # Proporção de dados para validação final (20% do total, 80% treino)
)

MOVING_AVERAGE_WINDOWS = [7, 14, 30]
FEATURES_CANDIDATAS = [
    "high",
    "low",
    "sma_7",
    "sma_14",
    "sma_30",
    "close_lag5",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_upper",
    "bb_lower",
    "bb_mavg",
    "daily_return",
    "volume",
    "buytakeramount",
    "buytakerquantity",
    "std7",
    "std14",
    "std30",
    "volatility_7d",
    "volatility_30d",
    "rsi",
    "obv",
]

INITIAL_INVESTMENT = 1000.0

LOG_LEVEL = "INFO"

USE_USD_BRL = True  # Se True, adiciona a cotação USD/BRL como feature obrigatória
