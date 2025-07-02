# data_loader.py
import pandas as pd
import logging
import urllib.error
from typing import Optional

def load_crypto_data(
    base_symbol: str,
    quote_symbol: str,
    timeframe: str,
    exchange: str = "Poloniex"
) -> Optional[pd.DataFrame]:
    """
    Carrega os dados históricos de um par de criptomoedas de uma exchange específica.
    Retorna um DataFrame do Pandas ou None em caso de falha.
    """
    pair = f"{base_symbol.upper()}{quote_symbol.upper()}"
    url = f"https://www.cryptodatadownload.com/cdd/{exchange.capitalize()}_{pair}_{timeframe}.csv"
    
    try:
        logging.info(f"Tentando carregar dados para {pair} de {url}")
        # O skiprows=1 pula o cabeçalho de aviso do site
        df = pd.read_csv(url, skiprows=1)

        if df.empty:
            logging.warning(f"Os dados para {pair} no timeframe '{timeframe}' estão vazios.")
            return None

        # Converte a coluna 'date' para datetime e ordena
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Assegura que a coluna 'close' é numérica
        if 'close' in df.columns:
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        return df

    except urllib.error.HTTPError as e:
        logging.error(f"FALHA ao carregar {pair}. O recurso não foi encontrado (Erro HTTP {e.code}).")
        return None
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado ao carregar {pair}. Erro: {e}")
        return None