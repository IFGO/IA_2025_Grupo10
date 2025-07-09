import pandas as pd
import numpy as np
import requests
from pathlib import Path

def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e pré-processa o DataFrame de dados brutos de criptomoedas.

    Esta função padroniza os nomes das colunas, converte a coluna de data
    para o formato datetime, remove colunas desnecessárias e elimina
    linhas com datas ausentes.

    Args:
        df: O DataFrame bruto para ser processado.

    Returns:
        O DataFrame processado, ordenado por data.

    Raises:
        ValueError: Se a coluna 'date' não for encontrada no DataFrame.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    if 'date' not in df.columns:
        raise ValueError(f"Coluna 'date' não encontrada. Colunas: {df.columns.tolist()}")
    # Converte a coluna 'date' para datetime, tratando erros
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    cols_to_drop = ['unix', 'symbol', 'tradecount']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    #Remove linhas que a coluna data é vazia, ordena por data e reseta o índice das linhas
    return df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

def calculate_financial_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calcula uma variedade de indicadores financeiros usando operações vetorizadas.

    Adiciona as seguintes colunas ao DataFrame:
    - daily_return: Retorno percentual diário.
    - moving_average: Média móvel simples do preço de fecho.
    - volatility: Desvio padrão dos retornos diários.
    - cumulative_return: Retorno cumulativo desde o início do período.
    - short_mavg, long_mavg: Médias móveis de curto e longo prazo.
    - signal: Sinal de negociação (1 para compra, -1 para venda, 0 para manter)
      baseado no cruzamento das médias móveis.

    Args:
        df: DataFrame contendo pelo menos a coluna 'close'.
        window: A janela (em dias) para calcular a média móvel e a volatilidade.

    Returns:
        O DataFrame original com as colunas de indicadores adicionadas.
    """
    #converte a coluna 'close' para numérico, tratando erros e removendo NaNs
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    # Calcula os indicadores financeiros
    df['daily_return'] = df['close'].pct_change()
    df['moving_average'] = df['close'].rolling(window=window).mean()
    df['volatility'] = df['daily_return'].rolling(window=window).std()
    df['cumulative_return'] = (1 + df['daily_return'].fillna(0)).cumprod()
    # Calcula as médias móveis de curto e longo prazo
    short_window = 10
    long_window = 30
    df['short_mavg'] = df['close'].rolling(window=short_window).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window).mean()
    
    previous_short_mavg = np.roll(df['short_mavg'], 1)
    previous_long_mavg = np.roll(df['long_mavg'], 1)
    # Cria o sinal de negociação baseado no cruzamento das médias móveis (+1 comprar, -1 vender, 0 manter)
    # np.roll é usado para obter o valor anterior sem perder o alinhamento do índice
    df['signal'] = np.where(
        (df['short_mavg'] > df['long_mavg']) & (previous_short_mavg <= previous_long_mavg), 
        1, 
        np.where(
            (df['short_mavg'] < df['long_mavg']) & (previous_short_mavg >= previous_long_mavg), 
            -1, 
            0
        )
    )
    
    return df

def load_crypto_data(
    base_symbol: str,
    quote_symbol: str,
    timeframe: str,
    exchange: str = "Poloniex",
    calculate_indicators: bool = True
) -> pd.DataFrame | None:
    """
    Carrega dados históricos de criptomoedas, fazendo o download se necessário.

    A função primeiro procura por um arquivo CSV local. Se não o encontrar,
    tenta fazer o download dos dados do CryptoDataDownload. Após carregar,
    pode opcionalmente calcular indicadores financeiros.

    Args:
        base_symbol: O símbolo da moeda base (ex: 'BTC').
        quote_symbol: O símbolo da moeda de cotação (ex: 'USDT').
        timeframe: O intervalo de tempo dos dados (ex: '1h', 'd').
        exchange: A exchange da qual obter os dados.
        calculate_indicators: Se True, calcula e adiciona indicadores financeiros
                              ao DataFrame.

    Returns:
        Um DataFrame do pandas com os dados processados e, opcionalmente,
        os indicadores, ou None se o carregamento falhar.
    """
    try:
        data_dir = Path("data/raw")
        filename_local = f"{base_symbol.upper()}_{quote_symbol.upper()}_{timeframe}.csv"
        filepath = data_dir / filename_local
        filename_remote = f"{base_symbol.upper()}{quote_symbol.upper()}_{timeframe}.csv"
        url = f"https://www.cryptodatadownload.com/cdd/{exchange}_{filename_remote}"
        
        # Faz o download do arquivo se não existir localmente
        if not filepath.exists():
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data_dir.mkdir(parents=True, exist_ok=True)
                    filepath.write_bytes(response.content)
                else:
                    return None
            except requests.RequestException:
                return None
        # Lê o arquivo CSV, verificando se a primeira linha é um cabeçalho
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            skip = 1 if not first_line.lower().startswith('date') else 0

        df = pd.read_csv(filepath, skiprows=skip, encoding='utf-8-sig')
        
        processed_df = _process_dataframe(df)

        if calculate_indicators and not processed_df.empty:
            return calculate_financial_indicators(processed_df)
            
        return processed_df

    except Exception:
        return None
