import pandas as pd
import numpy as np
from typing import List
import ta # Você precisaria instalar: pip install ta
import logging # Adicionado para warnings

# Adicionado para depuração:
# print("Loading src/feature_engineering.py")

def create_moving_average_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Cria features de média móvel e desvio padrão para a coluna 'close'.
    Não remove NaNs aqui; a remoção final será feita em create_technical_features.

    Args:
        df (pd.DataFrame): DataFrame de entrada com a coluna 'close'.
        windows (List[int]): Lista de tamanhos de janela para as médias móveis.

    Returns:
        pd.DataFrame: DataFrame com as novas features adicionadas.
    """
    df_featured = df.copy()
    for window in windows:
        # Garante que há dados suficientes para a janela
        if len(df_featured) >= window:
            df_featured[f'sma_{window}'] = df_featured['close'].rolling(window=window).mean()
            df_featured[f'std_{window}'] = df_featured['close'].rolling(window=window).std()
        else:
            # Se não há dados suficientes, preenche com NaN
            df_featured[f'sma_{window}'] = np.nan
            df_featured[f'std_{window}'] = np.nan
            logging.warning(f"DataFrame muito curto para calcular SMA/STD com janela {window}. Atribuindo NaN.")

    return df_featured # Removido .dropna() aqui

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features técnicas adicionais, incluindo retornos, volatilidade e indicadores técnicos.
    Realiza a remoção final de NaNs.

    Args:
        df (pd.DataFrame): DataFrame de entrada com as colunas 'close', 'high', 'low', 'open', 'volume'.
                           (Assumindo que essas colunas estão disponíveis no seu DataFrame de dados brutos)

    Returns:
        pd.DataFrame: DataFrame com as novas features técnicas adicionadas.
    """
    df_featured = df.copy()

    # Primeiro, adicione as features de média móvel e desvio padrão
    windows = [7, 14, 30] # Usando as mesmas janelas do main.py
    df_featured = create_moving_average_features(df_featured, windows) # Passa o df_featured para adicionar features

    # Retorno Diário
    df_featured['daily_return'] = df_featured['close'].pct_change()

    # Volatilidade (desvio padrão dos retornos em uma janela)
    # Certifique-se de que há dados suficientes para as janelas
    if len(df_featured) >= 7:
        df_featured['volatility_7d'] = df_featured['daily_return'].rolling(window=7).std() * np.sqrt(7)
    else:
        df_featured['volatility_7d'] = np.nan
        logging.warning("DataFrame muito curto para calcular volatility_7d.")

    if len(df_featured) >= 30:
        df_featured['volatility_30d'] = df_featured['daily_return'].rolling(window=30).std() * np.sqrt(30)
    else:
        df_featured['volatility_30d'] = np.nan
        logging.warning("DataFrame muito curto para calcular volatility_30d.")


    # Preço de fechamento do dia anterior (lag feature)
    df_featured['close_lag1'] = df_featured['close'].shift(1)
    df_featured['close_lag5'] = df_featured['close'].shift(5) # Exemplo de mais um lag

    # --- Adicionando Indicadores Técnicos (usando a biblioteca 'ta') ---
    # Certifique-se de que as colunas 'high', 'low', 'open', 'volume' existem no seu DataFrame
    # Se não existirem, você precisará adaptar ou remover essas features.

    required_cols_for_ta = ['open', 'high', 'low', 'close', 'volume']
    if all(col in df_featured.columns for col in required_cols_for_ta):
        # RSI (Relative Strength Index)
        if len(df_featured) >= 14: # RSI default window is 14
            df_featured['rsi'] = ta.momentum.RSIIndicator(close=df_featured['close'], window=14).rsi()
        else:
            df_featured['rsi'] = np.nan
            logging.warning("DataFrame muito curto para calcular RSI.")

        # MACD (Moving Average Convergence Divergence) - default fast=12, slow=26
        if len(df_featured) >= 26: # MACD needs at least 26 data points
            macd = ta.trend.MACD(close=df_featured['close'])
            df_featured['macd'] = macd.macd()
            df_featured['macd_signal'] = macd.macd_signal()
            df_featured['macd_diff'] = macd.macd_diff()
        else:
            df_featured['macd'] = np.nan
            df_featured['macd_signal'] = np.nan
            df_featured['macd_diff'] = np.nan
            logging.warning("DataFrame muito curto para calcular MACD.")

        # Bollinger Bands - default window=20
        if len(df_featured) >= 20:
            bollinger = ta.volatility.BollingerBands(close=df_featured['close'], window=20, window_dev=2)
            df_featured['bb_upper'] = bollinger.bollinger_hband()
            df_featured['bb_lower'] = bollinger.bollinger_lband()
            df_featured['bb_mavg'] = bollinger.bollinger_mavg()
        else:
            df_featured['bb_upper'] = np.nan
            df_featured['bb_lower'] = np.nan
            df_featured['bb_mavg'] = np.nan
            logging.warning("DataFrame muito curto para calcular Bollinger Bands.")

        # On-Balance Volume (OBV)
        # OBV não tem janela, mas pode ter NaNs se 'volume' ou 'close' tiverem
        df_featured['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df_featured['close'], volume=df_featured['volume']).on_balance_volume()
    else:
        logging.warning("Colunas 'open', 'high', 'low', 'volume' não encontradas para criar todas as features técnicas. Algumas features serão omitidas. Colunas disponíveis: %s", df_featured.columns.tolist())

    df_featured = df_featured.dropna() # Alterado para não usar inplace=True
    return df_featured
