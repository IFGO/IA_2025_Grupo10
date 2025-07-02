import pandas as pd
import os

def load_crypto_data(base_symbol: str, quote_symbol: str, timeframe: str, exchange: str = "Poloniex") -> pd.DataFrame:
    try:
        filename = f"{base_symbol.upper()}_{quote_symbol.upper()}_{timeframe}.csv"
        filepath = os.path.join("data", "raw", filename)

        print(f"[DEBUG] Verificando caminho: {filepath} | Existe? {os.path.exists(filepath)}")

        if not os.path.exists(filepath):
            return None

        # ← skip a linha de comentário e trata BOM invisível
        df = pd.read_csv(filepath, skiprows=1, encoding='utf-8-sig')

        # Mostrar as colunas reais lidas
        print("[DEBUG] Colunas lidas do CSV:", df.columns.tolist())

        # Padronizar nomes
        df.columns = [col.strip().lower() for col in df.columns]

        print("[DEBUG] Colunas normalizadas:", df.columns.tolist())

        # Garantir que 'date' exista
        if 'date' not in df.columns:
            raise ValueError(f"[ERRO] Coluna 'date' não encontrada. Colunas disponíveis: {df.columns.tolist()}")

        df['date'] = pd.to_datetime(df['date'])

        return df.sort_values('date')

    except Exception as e:
        print(f"[ERRO] Falha ao carregar dados de {base_symbol}: {e}")
        return None
