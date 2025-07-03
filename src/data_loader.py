import pandas as pd
import os
import requests

def load_crypto_data(base_symbol: str, quote_symbol: str, timeframe: str, exchange: str = "Poloniex") -> pd.DataFrame:
    try:
        filename = f"{base_symbol.upper()}_{quote_symbol.upper()}_{timeframe}.csv"
        filepath = os.path.join("data", "raw", filename)

        print(f"[DEBUG] Verificando caminho: {filepath} | Existe? {os.path.exists(filepath)}")

        if not os.path.exists(filepath):
            # Monta a URL para download
            url = f"https://www.cryptodatadownload.com/cdd/{exchange}_{base_symbol.upper()}{quote_symbol.upper()}_{timeframe}.csv"
            print(f"[DEBUG] Tentando baixar arquivo de: {url}")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"[DEBUG] Download concluído e salvo em: {filepath}")
                else:
                    print(f"[ERRO] Falha ao baixar arquivo. Status code: {response.status_code}")
                    return None
            except Exception as e:
                print(f"[ERRO] Exceção ao tentar baixar arquivo: {e}")
                return None

        # escapa a linha de comentário e trata BOM invisível
        df = pd.read_csv(filepath, skiprows=1, encoding='utf-8-sig')

        # Mostrar as colunas reais lidas
        print("[DEBUG] Colunas lidas do CSV:", df.columns.tolist())

        # Padronizar nomes (minúsculo, sem espaços, troca espaços por underline)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        print("[DEBUG] Colunas normalizadas:", df.columns.tolist())

        # Remover colunas não numéricas e irrelevantes, se existirem
        for col in ['symbol', 'exchange', 'unix']:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Remover linhas onde 'close' não é numérico (ex: header residual ou erro de leitura)
        if 'close' in df.columns:
            df = df[pd.to_numeric(df['close'], errors='coerce').notnull()]
            df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Garantir que 'date' exista
        if 'date' not in df.columns:
            raise ValueError(f"[ERRO] Coluna 'date' não encontrada. Colunas disponíveis: {df.columns.tolist()}")

        df['date'] = pd.to_datetime(df['date'])

        return df.sort_values('date')

    except Exception as e:
        print(f"[ERRO] Falha ao carregar dados de {base_symbol}: {e}")
        return None
