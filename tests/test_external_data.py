
import pytest
from datetime import datetime, timedelta
from src.external_data import fetch_usd_brl_bacen

def test_fetch_usd_brl_bacen_range():
    start_date = "2019-01-01"
    end_date = "2019-01-10"
    df = fetch_usd_brl_bacen(start_date, end_date)

    # Verifica se DataFrame não está vazio
    assert not df.empty, "O DataFrame retornado está vazio"

    # Verifica se as colunas esperadas estão presentes
    assert "date" in df.columns, "Coluna 'date' ausente"
    assert "usd_brl" in df.columns, "Coluna 'usd_brl' ausente"

    # Verifica se as datas estão no intervalo correto
    assert df["date"].min() >= datetime.strptime(start_date, "%Y-%m-%d")
    assert df["date"].max() <= datetime.strptime(end_date, "%Y-%m-%d")
