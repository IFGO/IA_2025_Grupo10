# src/feature_engineering.py

import pandas as pd
import numpy as np
from typing import List

# Preste atenção neste nome! É este que deve ser importado.
def create_moving_average_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Cria features de média móvel e desvio padrão.
    """
    df_featured = df.copy()
    for window in windows:
        df_featured[f'sma_{window}'] = df_featured['close'].rolling(window=window).mean()
        df_featured[f'std_{window}'] = df_featured['close'].rolling(window=window).std()

    df_featured.dropna(inplace=True)
    return df_featured

# A função 'create_technical_features' pode não existir ou ter outro nome.