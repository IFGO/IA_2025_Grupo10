import pandas as pd
import requests
import logging
from datetime import datetime, timedelta

def fetch_usd_brl_bacen(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Busca a série temporal da cotação de venda USD/BRL do Banco Central.

    A função lida automaticamente com pedidos para períodos superiores a 10
    anos, quebrando a consulta em múltiplos pedidos à API para contornar
    as limitações do serviço.

    Args:
        start_date (str): A data de início para a consulta, no formato "YYYY-MM-DD".
        end_date (str): A data de fim para a consulta, no formato "YYYY-MM-DD".

    Returns:
        pd.DataFrame: Um DataFrame contendo as colunas 'date' e 'usd_brl'
                      com os dados da cotação. Retorna um DataFrame vazio
                      em caso de falha na comunicação com a API.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

        all_data = []

        while start_dt < end_dt:
            block_end = min(start_dt + timedelta(days=3652), end_dt)

            start_fmt = start_dt.strftime("%d/%m/%Y")
            end_fmt   = block_end.strftime("%d/%m/%Y")

            url = (
                "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
                f"?formato=json&dataInicial={start_fmt}&dataFinal={end_fmt}"
            )

            headers = {"Accept": "application/json"}
            response = requests.get(url, headers=headers,timeout=30)
            response.raise_for_status()
            data = response.json()

            if data:
                df = pd.DataFrame(data)
                all_data.append(df)

            start_dt = block_end + timedelta(days=1)
        if not all_data:
            logging.warning("Nenhum dado encontrado para o período especificado.")
            return pd.DataFrame()

        df_all = pd.concat(all_data, ignore_index=True)
        df_all["date"] = pd.to_datetime(df_all["data"], format="%d/%m/%Y")
        df_all["usd_brl"] = df_all["valor"].str.replace(",", ".").astype(float)

        return df_all[["date", "usd_brl"]].sort_values("date")

    except Exception as e:
        print(f"[ERRO] Falha ao buscar USD/BRL do BACEN: {e}")
        return pd.DataFrame()
