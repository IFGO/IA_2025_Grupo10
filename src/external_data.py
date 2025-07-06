import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_usd_brl_bacen(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Obtém a cotação diária USD/BRL (venda) do BACEN (Série 1), mesmo para períodos maiores que 10 anos.
    Divide automaticamente em blocos válidos de até 10 anos.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

        # Lista para armazenar blocos
        all_data = []

        # Loop por blocos de no máximo 10 anos (3652 dias)
        while start_dt < end_dt:
            block_end = min(start_dt + timedelta(days=3652), end_dt)

            start_fmt = start_dt.strftime("%d/%m/%Y")
            end_fmt   = block_end.strftime("%d/%m/%Y")

            url = (
                "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
                f"?formato=json&dataInicial={start_fmt}&dataFinal={end_fmt}"
            )

            headers = {"Accept": "application/json"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data)
            all_data.append(df)

            # Avança para o próximo bloco
            start_dt = block_end + timedelta(days=1)

        # Concatena todos os blocos em um único DataFrame
        df_all = pd.concat(all_data, ignore_index=True)
        df_all["date"] = pd.to_datetime(df_all["data"], format="%d/%m/%Y")
        df_all["usd_brl"] = df_all["valor"].str.replace(",", ".").astype(float)

        return df_all[["date", "usd_brl"]].sort_values("date")

    except Exception as e:
        print(f"[ERRO] Falha ao buscar USD/BRL do BACEN: {e}")
        return pd.DataFrame()
