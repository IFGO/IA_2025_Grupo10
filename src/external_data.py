# -*- coding: utf-8 -*-
"""
Coleta de Dados Financeiros de Fontes Externas.

Este módulo contém funções para buscar dados financeiros de APIs públicas,
como a do Banco Central do Brasil (BACEN). O objetivo é encapsular a lógica
de comunicação com essas fontes, tratando da paginação, formatação dos dados
e tratamento de erros, e entregando um DataFrame limpo e pronto para uso.

A principal funcionalidade implementada é a busca da série histórica da cotação
de venda do dólar (USD/BRL).
"""
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta


def fetch_usd_brl_bacen(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Busca a série temporal da cotação de venda USD/BRL do Banco Central do Brasil.

    Esta função consulta a API de Séries Temporais do BACEN para obter a cotação
    diária de venda do dólar. Ela foi projetada para lidar com períodos de
    consulta longos (superiores a 10 anos), quebrando a requisição em múltiplos
    pedidos menores para contornar as limitações do serviço da API.

    Args:
        start_date (str): A data de início para a consulta, no formato "YYYY-MM-DD".
        end_date (str): A data de fim para a consulta, no formato "YYYY-MM-DD".

    Returns:
        pd.DataFrame: Um DataFrame do Pandas contendo as colunas 'date' (em formato
                      datetime) e 'usd_brl' (cotação de venda como float). Retorna
                      um DataFrame vazio em caso de falha na comunicação com a API
                      ou se nenhum dado for encontrado para o período.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = []

        while start_dt < end_dt:
            block_end = min(start_dt + timedelta(days=3652), end_dt)

            start_fmt = start_dt.strftime("%d/%m/%Y")
            end_fmt = block_end.strftime("%d/%m/%Y")

            url = (
                "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
                f"?formato=json&dataInicial={start_fmt}&dataFinal={end_fmt}"
            )

            headers = {"Accept": "application/json"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data:
                df = pd.DataFrame(data)
                all_data.append(df)  # type: ignore

            start_dt = block_end + timedelta(days=1)
        if not all_data:
            logging.warning("Nenhum dado encontrado para o período especificado.")
            return pd.DataFrame()

        df_all = pd.concat(all_data, ignore_index=True)  # type: ignore
        df_all["date"] = pd.to_datetime(df_all["data"], format="%d/%m/%Y")  # type: ignore
        df_all["usd_brl"] = df_all["valor"].str.replace(",", ".").astype(float)  # type: ignore

        return df_all[["date", "usd_brl"]].sort_values("date")  # type: ignore

    except Exception as e:
        print(f"[ERRO] Falha ao buscar USD/BRL do BACEN: {e}")
        return pd.DataFrame()
