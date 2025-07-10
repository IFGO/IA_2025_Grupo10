# -*- coding: utf-8 -*-
"""
Engenharia de Features para Dados Financeiros de Criptoativos.

Este módulo fornece um conjunto de funções para criar e adicionar novas features
(características) a um DataFrame de dados de séries temporais financeiras. O
objetivo é enriquecer os dados brutos com informações que possam melhorar o
desempenho de modelos de machine learning ou análises quantitativas.

As funcionalidades incluem:
-   **Enriquecimento com Dados Externos:** Adição de indicadores macroeconômicos,
    como a taxa de câmbio USD/BRL.
-   **Features de Médias Móveis:** Cálculo de médias móveis simples e desvios
    padrão para diferentes janelas de tempo.
-   **Indicadores de Análise Técnica:** Um pipeline completo que utiliza a
    biblioteca 'ta' para calcular um vasto conjunto de indicadores, como:
    -   Volatilidade (baseada nos retornos diários).
    -   Features de Lag (preços de dias anteriores).
    -   Índice de Força Relativa (RSI).
    -   Convergência e Divergência de Médias Móveis (MACD).
    -   Bandas de Bollinger (Bollinger Bands).
    -   On-Balance Volume (OBV).

O módulo foi projetado para ser modular, permitindo que as funções de criação
de features sejam combinadas conforme a necessidade.
"""
import pandas as pd
import numpy as np
from typing import List
import ta
import logging 
from src.external_data import fetch_usd_brl_bacen


def enrich_with_external_features(df: pd.DataFrame, use_usd_brl: bool = True) -> pd.DataFrame:
    """
    Enriquece o DataFrame com dados macroeconômicos externos.

    Atualmente, a função adiciona a cotação da taxa de câmbio USD/BRL, obtida
    através da API do Banco Central do Brasil (BACEN). Os dados são unidos
    ao DataFrame original com base na coluna 'date'.

    Args:
        df (pd.DataFrame): O DataFrame principal a ser enriquecido. Deve conter
                           uma coluna 'date' para a fusão dos dados.
        use_usd_brl (bool, optional): Controla se a cotação USD/BRL deve ser
                                      adicionada. Padrão é True.

    Returns:
        pd.DataFrame: O DataFrame original com os dados externos adicionados.
                      Se a busca dos dados externos falhar, retorna o
                      DataFrame original com um aviso.
    """
    if use_usd_brl:
        start = df["date"].min().strftime("%Y-%m-%d")
        end = df["date"].max().strftime("%Y-%m-%d")
        usd_brl_df = fetch_usd_brl_bacen(start, end)

        if not usd_brl_df.empty:
            df = pd.merge(df, usd_brl_df, on="date", how="left")
            if 'usd_brl' in df.columns:
                df['usd_brl'] = df['usd_brl'].astype(np.float32)
        else:
            logging.warning("Cotação USD/BRL não foi adicionada (dados indisponíveis).")

    return df


def create_moving_average_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Cria features baseadas em médias móveis para uma lista de janelas de tempo.

    Para cada janela especificada na lista, esta função calcula a Média Móvel
    Simples (SMA) e o Desvio Padrão (STD) do preço de fechamento ('close').

    Args:
        df (pd.DataFrame): O DataFrame de entrada, que deve conter a coluna 'close'.
        windows (List[int]): Uma lista de inteiros representando os tamanhos
                             das janelas para os cálculos (ex: [7, 14, 30]).

    Returns:
        pd.DataFrame: Uma cópia do DataFrame com as novas colunas de SMA e STD
                      adicionadas para cada janela.
    """
    df_featured = df.copy()
    for window in windows:
        if len(df_featured) >= window:
            df_featured[f'sma_{window}'] = df_featured['close'].rolling(window=window).mean()
            df_featured[f'std_{window}'] = df_featured['close'].rolling(window=window).std()
        else:
            df_featured[f'sma_{window}'] = np.nan
            df_featured[f'std_{window}'] = np.nan
            logging.warning(f"DataFrame muito curto para calcular SMA/STD com janela {window}. Atribuindo NaN.")

    return df_featured

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """"
    Adiciona um conjunto abrangente de features de análise técnica ao DataFrame.

    Esta função atua como um pipeline principal para a engenharia de features,
    adicionando médias móveis, retornos, volatilidade, lags de preço e
    indicadores técnicos complexos da biblioteca 'ta'. Ao final, remove todas
    as linhas que contenham valores NaN resultantes dos cálculos com janelas.

    Args:
        df (pd.DataFrame): O DataFrame de entrada. Requer as colunas 'open',
                           'high', 'low', 'close', e uma coluna de volume
                           (ex: 'volume' ou 'volume_usdt').

    Returns:
        pd.DataFrame: O DataFrame enriquecido com dezenas de novas features
                      técnicas, e sem linhas com valores ausentes.
    """
    df_featured = df.copy()

    windows = [7, 14, 30] 
    df_featured = create_moving_average_features(df_featured, windows) 
    
    df_featured['daily_return'] = df_featured['close'].pct_change()

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


    df_featured['close_lag1'] = df_featured['close'].shift(1)
    df_featured['close_lag5'] = df_featured['close'].shift(5) 

    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df_featured.columns]

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
        if len(df_featured) >= 14:
            df_featured['rsi'] = ta.momentum.RSIIndicator(close=df_featured['close'], window=14).rsi()
        else:
            df_featured['rsi'] = np.nan
            logging.warning("DataFrame muito curto para calcular RSI.")
        
        if len(df_featured) >= 26:
            macd = ta.trend.MACD(close=df_featured['close'])
            df_featured['macd'] = macd.macd()
            df_featured['macd_signal'] = macd.macd_signal()
            df_featured['macd_diff'] = macd.macd_diff()
        else:
            df_featured['macd'] = np.nan
            df_featured['macd_signal'] = np.nan
            df_featured['macd_diff'] = np.nan
            
        if len(df_featured) >= 20:
            boll = ta.volatility.BollingerBands(close=df_featured['close'], window=20)
            df_featured['bb_upper'] = boll.bollinger_hband()
            df_featured['bb_lower'] = boll.bollinger_lband()
            df_featured['bb_mavg'] = boll.bollinger_mavg()
        else:
            df_featured['bb_upper'] = df_featured['bb_lower'] = df_featured['bb_mavg'] = np.nan            

        df_featured['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df_featured['close'],
            volume=df_featured[volume_col]
        ).on_balance_volume()

    df_featured = df_featured.dropna()
    return df_featured
