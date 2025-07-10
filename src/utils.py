# -*- coding: utf-8 -*-
"""
Módulo de Funções Utilitárias.

Este módulo contém funções auxiliares e de conveniência que são usadas em
várias partes do projeto. O objetivo é centralizar lógica comum, como a
geração de nomes de arquivos e caminhos, para evitar duplicação de código e
melhorar a manutenibilidade.
"""
import os
from config import (
    MOEDA_COTACAO, TIMEFRAME, OUTPUT_FOLDER, PROCESSED_DATA_FOLDER,
    MODELS_FOLDER, RAW_FILENAME_TEMPLATE, FEATURED_FILENAME_TEMPLATE,
    MODEL_FILENAME_TEMPLATE
)

def get_pair_key(base_symbol: str) -> str:
    """Gera a chave padronizada para um par (ex: 'BTC_USDT')."""
    return f"{base_symbol.upper()}_{MOEDA_COTACAO.upper()}"

def get_raw_data_filepath(base_symbol: str) -> str:
    """
    Monta o caminho completo para o arquivo de dados brutos.

    Args:
        base_symbol (str): O símbolo da criptomoeda base (ex: 'BTC').

    Returns:
        str: O caminho absoluto para o arquivo .csv de dados brutos.
    """
    filename = RAW_FILENAME_TEMPLATE.format(
        base=base_symbol.upper(),
        quote=MOEDA_COTACAO.upper(),
        timeframe=TIMEFRAME
    )
    return os.path.join(OUTPUT_FOLDER, filename)

def get_processed_data_filepath(base_symbol: str) -> str:
    """
    Monta o caminho completo para o arquivo de dados com features.

    Args:
        base_symbol (str): O símbolo da criptomoeda base (ex: 'BTC').

    Returns:
        str: O caminho absoluto para o arquivo .csv de dados processados.
    """
    filename = FEATURED_FILENAME_TEMPLATE.format(
        base=base_symbol.upper(),
        quote=MOEDA_COTACAO.upper()
    )
    return os.path.join(PROCESSED_DATA_FOLDER, filename)

def get_model_filepath(model_type: str, base_symbol: str) -> str:
    """
    Monta o caminho completo para o arquivo de modelo salvo.

    Args:
        model_type (str): O tipo do modelo (ex: 'mlp', 'randomforest').
        base_symbol (str): O símbolo da criptomoeda base (ex: 'BTC').

    Returns:
        str: O caminho absoluto para o arquivo .pkl do modelo.
    """
    filename = MODEL_FILENAME_TEMPLATE.format(
        model_type=model_type.lower(),
        base=base_symbol.upper(),
        quote=MOEDA_COTACAO.upper()
    )
    return os.path.join(MODELS_FOLDER, filename)