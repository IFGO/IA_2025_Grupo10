import pandas as pd
import requests
import logging
from pathlib import Path

def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função auxiliar para limpar, padronizar e processar o DataFrame.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    if 'date' not in df.columns:
        raise ValueError(f"Coluna 'date' não encontrada. Colunas: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    cols_to_drop = ['unix', 'symbol', 'tradecount']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df.dropna(subset=['date']).sort_values('date')

def load_crypto_data(
    base_symbol: str, 
    quote_symbol: str, 
    timeframe: str, 
    exchange: str = "Poloniex"
) -> pd.DataFrame | None:
    """
    Carrega dados de criptomoedas de um arquivo local ou faz o download,
    aplicando as melhores práticas de organização e manipulação de arquivos.
    """
    try:
        filename = f"{base_symbol.upper()}{quote_symbol.upper()}_{timeframe}.csv"
        data_dir = Path("data/raw")
        filepath = data_dir / filename

        if not filepath.exists():
            logging.info(f"Arquivo '{filepath}' não encontrado. Tentando download...")
            url = f"https://www.cryptodatadownload.com/cdd/{exchange}_{filename}"
            
            try:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    data_dir.mkdir(parents=True, exist_ok=True)
                    filepath.write_bytes(response.content)
                    logging.info(f"Download concluído com sucesso.")
                else:
                    logging.error(f"Falha no download. Servidor retornou status: {response.status_code}")
                    return None
            except requests.RequestException as e:
                logging.error(f"Exceção de rede durante o download: {e}")
                return None

        logging.debug(f"Lendo arquivo: {filepath}")
        df = pd.read_csv(filepath, skiprows=1, encoding='utf-8-sig')
        
        return _process_dataframe(df)

    except Exception as e:
        logging.error(f"Falha crítica ao carregar/processar dados para {base_symbol}: {e}")
        return None