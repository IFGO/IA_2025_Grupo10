import pandas as pd
import numpy as np
from typing import List
import ta

def create_moving_average_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Cria features de média móvel e desvio padrão para a coluna 'close'.

    Args:
        df (pd.DataFrame): DataFrame de entrada com a coluna 'close'.
        windows (List[int]): Lista de tamanhos de janela para as médias móveis.

    Returns:
        pd.DataFrame: DataFrame com as novas features adicionadas.
    """
    df_featured = df.copy()
    for window in windows:
        df_featured[f'sma_{window}'] = df_featured['close'].rolling(window=window).mean()
        df_featured[f'std_{window}'] = df_featured['close'].rolling(window=window).std()

    # Remove as linhas que contêm NaN devido ao cálculo das médias móveis iniciais
    df_featured.dropna(inplace=True)
    return df_featured

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features técnicas adicionais, incluindo retornos, volatilidade e indicadores técnicos.

    Args:
        df (pd.DataFrame): DataFrame de entrada com as colunas 'close', 'high', 'low', 'open', 'volume'.
                           (Assumindo que essas colunas estão disponíveis no seu DataFrame de dados brutos)

    Returns:
        pd.DataFrame: DataFrame com as novas features técnicas adicionadas.
    """
    df_featured = df.copy()

    # Retorno Diário
    df_featured['daily_return'] = df_featured['close'].pct_change()

    # Volatilidade (desvio padrão dos retornos em uma janela)
    df_featured['volatility_7d'] = df_featured['daily_return'].rolling(window=7).std() * np.sqrt(7)
    df_featured['volatility_30d'] = df_featured['daily_return'].rolling(window=30).std() * np.sqrt(30)

    # Preço de fechamento do dia anterior (lag feature)
    df_featured['close_lag1'] = df_featured['close'].shift(1)
    df_featured['close_lag5'] = df_featured['close'].shift(5) # Exemplo de mais um lag

    # --- Adicionando Indicadores Técnicos (usando a biblioteca 'ta') ---
    # Certifique-se de que as colunas 'high', 'low', 'open', 'volume' existem no seu DataFrame
    # Se não existirem, você precisará adaptar ou remover essas features.

    if all(col in df_featured.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        # RSI (Relative Strength Index)
        df_featured['rsi'] = ta.momentum.RSIIndicator(close=df_featured['close'], window=14).rsi()

        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close=df_featured['close'])
        df_featured['macd'] = macd.macd()
        df_featured['macd_signal'] = macd.macd_signal()
        df_featured['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df_featured['close'], window=20, window_dev=2)
        df_featured['bb_upper'] = bollinger.bollinger_hband()
        df_featured['bb_lower'] = bollinger.bollinger_lband()
        df_featured['bb_mavg'] = bollinger.bollinger_mavg()

        # On-Balance Volume (OBV)
        df_featured['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df_featured['close'], volume=df_featured['volume']).on_balance_volume()
    else:
        logging.warning("Colunas 'open', 'high', 'low', 'volume' não encontradas para criar todas as features técnicas. Algumas features serão omitidas.")


    df_featured.dropna(inplace=True)
    return df_featured