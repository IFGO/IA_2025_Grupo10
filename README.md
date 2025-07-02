Projeto de previsão de preços de criptomoedas com base em aprendizado de máquina, desenvolvido como trabalho final do Módulo I da Especialização em Inteligência Artificial Aplicada (2025).

## Estrutura

```
.
├── data/                      # Arquivos CSV com os dados das criptomoedas
├── figures/                   # Gráficos gerados automaticamente
├── models.py                  # Lógica de treinamento e simulação
├── data_load.py               # Carregamento e pré-processamento de dados
├── statistics_module.py       # Análises estatísticas e gráficos
├── main.py                    # Interface de linha de comando
├── tests/                     # Testes automatizados com pytest
├── requirements.txt
└── README.md
```

## Instalação

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Uso

### Análise Estatística:
```bash
python main.py stats --cryptos AAVEBTC LTCBTC ADABTC
```

### Treinamento de modelo:
```bash
python main.py train --crypto AAVEBTC --model mlp
```

### Simulação de investimento:
```bash
python main.py simulate --crypto AAVEBTC --model mlp --initial-investment 1000
```

## Testes

Execute os testes com cobertura de código:

```bash
pytest --cov=. --cov-report=term
```

## Observações

- Os arquivos CSV devem estar na pasta `data/`, com nomes no formato `Poloniex_{SYMBOL}_d.csv`.
- Os gráficos são salvos automaticamente em `figures/` com resolução mínima de 150 dpi.

---

Desenvolvido por: Thales Augusto Salvador, Carlos Henrique, Miguel Toledo(2025)
