import pandas as pd
import numpy as np
from typing import List
import ta # Você precisaria instalar: pip install ta
import logging # Adicionado para warnings
from src.external_data import fetch_usd_brl_bacen


def enrich_with_external_features(df: pd.DataFrame, use_usd_brl: bool = True) -> pd.DataFrame:
    """
    Adiciona variáveis macroeconômicas externas ao dataframe principal.
    Atualmente: cotação USD/BRL via BACEN.
    """
    if use_usd_brl:
        start = df["date"].min().strftime("%Y-%m-%d")
        end = df["date"].max().strftime("%Y-%m-%d")
        usd_brl_df = fetch_usd_brl_bacen(start, end)

        if not usd_brl_df.empty:
            df = pd.merge(df, usd_brl_df, on="date", how="left")
        else:
            print("[AVISO] Cotação USD/BRL não foi adicionada (dados indisponíveis).")

    return df


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

    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df_featured.columns]

    # Procurar coluna de volume com nomes alternativos
    volume_col = None
    for candidate in ['volume', 'volume_eth', 'volume_usdt']:
        if candidate in df_featured.columns:
            volume_col = candidate
            break

    if missing_cols or volume_col is None:
        logging.warning(
            "Colunas ausentes para cálculo técnico: %s. Colunas de volume disponíveis: %s. Colunas no DataFrame: %s",
            missing_cols,
            [col for col in df_featured.columns if 'volume' in col],
            df_featured.columns.tolist()
        )
    else:
        # RSI
        if len(df_featured) >= 14:
            df_featured['rsi'] = ta.momentum.RSIIndicator(close=df_featured['close'], window=14).rsi()
        else:
            df_featured['rsi'] = np.nan
            logging.warning("DataFrame muito curto para calcular RSI.")

        # MACD
        if len(df_featured) >= 26:
            macd = ta.trend.MACD(close=df_featured['close'])
            df_featured['macd'] = macd.macd()
            df_featured['macd_signal'] = macd.macd_signal()
            df_featured['macd_diff'] = macd.macd_diff()
        else:
            df_featured['macd'] = np.nan
            df_featured['macd_signal'] = np.nan
            df_featured['macd_diff'] = np.nan
            logging.warning("DataFrame muito curto para calcular MACD.")

        # Bollinger Bands
        if len(df_featured) >= 20:
            boll = ta.volatility.BollingerBands(close=df_featured['close'], window=20)
            df_featured['bb_upper'] = boll.bollinger_hband()
            df_featured['bb_lower'] = boll.bollinger_lband()
            df_featured['bb_mavg'] = boll.bollinger_mavg()
        else:
            df_featured['bb_upper'] = df_featured['bb_lower'] = df_featured['bb_mavg'] = np.nan
            logging.warning("DataFrame muito curto para calcular Bollinger Bands.")

        # OBV
        df_featured['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df_featured['close'],
            volume=df_featured[volume_col]
        ).on_balance_volume()

    df_featured = df_featured.dropna() # Alterado para não usar inplace=True
    return df_featured
