Projeto de previsão de preços de criptomoedas com base em aprendizado de máquina, desenvolvido como trabalho final do Módulo I da Especialização em Inteligência Artificial Aplicada (2025).

## Estrutura

```
.
├── data/
│   ├── raw/                        # Dados brutos baixados das exchanges
│   └── processed/                  # Dados após a engenharia de features
├── figures/                        # Onde todos os gráficos e relatórios visuais são salvos
│   ├── simple_plots/               # Gráficos de séries temporais simples
│   ├── analysis_plots/             # Gráficos de análise estatística (histogramas, boxplots, diagramas de dispersão)
│   ├── profit_plots/               # Gráficos de evolução do lucro das simulações
│   └── statistical_reports/        # Relatórios de testes estatísticos (hipótese, ANOVA)
├── models/                         # Modelos de Machine Learning treinados (salvos como arquivos .pkl)
├── src/                            # Módulos Python principais com a lógica do negócio
│   ├── init.py                     # Torna 'src' um pacote Python
│   ├── data_loader.py              # Responsável por carregar dados históricos de criptomoedas
│   ├── data_analyzer.py            # Realiza análises estatísticas descritivas e gera gráficos de análise
│   ├── data_visualizer.py          # Gera gráficos de linha simples para séries temporais
│   ├── feature_engineering.py      # Cria e transforma features para os modelos
│   ├── model_training.py           # Lida com o treinamento, avaliação e comparação de modelos de regressão
│   ├── prediction_profit.py        # Simula investimentos e calcula o lucro potencial
│   └── statistical_tests.py        # Implementa testes de hipótese e Análise de Variância (ANOVA)
├── tests/                          # Casos de teste automatizados para validação do código
│   ├── init.py                     # Torna 'tests' um pacote Python
│   ├── test_data_loader.py         #Teste do data loader (carregador dos dados históricos)
│   ├── test_data_analyzer.py       #Teste do data_analyzer (analisador estatístico)
│   ├── test_data_visualizer.py     #Teste do data_visualizer (gerador de gráficos pra séries temporais)
│   ├── test_feature_engineering.py #Teste do feature_engineering (criador e transformador de features para os modelos)
│   └── test_model_training.py      #Teste do model_training (treinador, avaliador e comparativo de modelos de regressão)
├── choose_var_training/            # Scripts auxiliares para análise e seleção de variáveis de treino
│   ├── escolher_variaveis_treino.py# Gera heatmaps de correlação entre variáveis dos arquivos processados
│   └── otimizando_variaveis.py     # Ajusta e avalia modelos de regressão linear múltipla para seleção de variáveis
├── main.py                         # Script principal configurável via linha de comando (CLI)
├── config.py                       # (Opcional) Arquivo para configurações globais do projeto
├── README.md                       # Este arquivo de documentação
└── requirements.txt                # Lista de dependências do projeto
```
**Clone o repositório (se aplicável) ou crie a estrutura de pastas:**

```bash
# Se você estiver clonando de um repositório Git
git clone <https://github.com/carlos-nitidum/projeto_cripto_ia>
cd projeto_cripto_ia
```

## Instalação

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
O arquivo `requirements.txt` indica os requisitos que o projeto deve conter, que atualmente são:
    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    statsmodels
    pytest
    pytest-cov
    black
    ruff
    python-dotenv
    pytest
    pytest-cov
    ta
    requests
    ```
## Uso

## Como Executar

O script principal `main.py` é configurável via linha de comando (CLI) usando `argparse`.

### Exemplos de Uso:

* **Executar todo o fluxo (download, análise, features, treinamento, lucro, estatísticas) para todas as criptomoedas configuradas:**
    ```bash
    python main.py --action all
    ```
* **Executar todo o fluxo integrndo a cotação oficial do dólar (USD/BRL) como feature externa para enriquecer os dados das criptomoedas, para todas as criptomoedas configuradas:**

    ```bash
    python main.py --action all --use_usd_brl
    ```

* **Apenas baixar os dados para todas as criptomoedas:**
    ```bash
    python main.py --action download
    ```

* **Apenas realizar a análise estatística e gerar gráficos para o Bitcoin (BTC):**
    ```bash
    python main.py --action analyze --crypto BTC
    ```

* **Realizar a engenharia de features para o Ethereum (ETH):**
    ```bash
    python main.py --action features --crypto ETH
    ```

* **Treinar o modelo MLP para todas as criptomoedas com 10 folds para validação cruzada:**
    ```bash
    python main.py --action train --model MLP --kfolds 10
    ```

* **Treinar o modelo de Regressão Polinomial (grau 3) para o XRP:**
    ```bash
    python main.py --action train --model Polynomial --poly_degree 3 --crypto XRP
    ```

* **Simular o lucro para o Litecoin (LTC):**
    ```bash
    python main.py --action profit --crypto LTC
    ```

* **Realizar testes estatísticos (teste de hipótese e ANOVA) para todas as criptomoedas, com um retorno alvo de 0.02% para o teste de hipótese:**
    ```bash
    python main.py --action stats --target_return_percent 0.0002
    ```
    (Note que `--target_return_percent` é um valor decimal, por exemplo, `0.0002` representa `0.02%`).

### Parâmetros Disponíveis:

* `--action`: Define a etapa do fluxo de trabalho a ser executada.
    * `all` (padrão): Executa todas as etapas em sequência.
    * `download`: Apenas baixa os dados brutos.
    * `analyze`: Realiza análises estatísticas descritivas e gera gráficos.
    * `features`: Realiza a engenharia de features nos dados.
    * `train`: Treina, avalia e compara os modelos de previsão.
    * `profit`: Simula o investimento e calcula o lucro obtido pelos modelos.
    * `stats`: Realiza testes de hipótese e análises ANOVA.
* `--crypto`: Símbolo da criptomoeda para processar (ex: `BTC`, `ETH`). Use `all` (padrão) para aplicar a ação a todas as criptomoedas configuradas internamente no `main.py`.
* `--model`: Tipo de modelo a ser usado para treinamento (aplicável com `--action train`).
    * `MLP` (padrão): Multi Layer Perceptron (Rede Neural).
    * `Linear`: Regressão Linear.
    * `Polynomial`: Regressão Polinomial.
    * `RandomForest`: Random Forest Regressor.
* `--kfolds`: Número de folds para K-fold cross-validation (padrão: `5`).
* `--target_return_percent`: O valor percentual (em decimal) para o teste de hipótese (padrão: `0.01` para `1%`).
* `--poly_degree`: Grau máximo para a regressão polinomial (de `2` a `10`, padrão: `2`).

## Executando Testes Automatizados

Para garantir a qualidade e a robustez do código, o projeto inclui testes unitários automatizados.

1.  Certifique-se de que seu ambiente virtual está ativado.
2.  Navegue até a pasta raiz do projeto (`projeto_cripto_ia`).
3.  Execute o `pytest` com o `pytest-cov` para gerar um relatório de cobertura:
    ```bash
    pytest --cov=src tests/
    ```
    Este comando executará todos os testes na pasta `tests/` e mostrará a porcentagem de cobertura do código na pasta `src/`.

    Para um relatório de cobertura HTML detalhado (visualizável no navegador):
    ```bash
    pytest --cov=src --cov-report=html tests/
    ```
    Isso criará uma pasta `htmlcov/` na raiz do projeto. Abra `htmlcov/index.html` em seu navegador.

## Boas Práticas de Código

O projeto segue as seguintes boas práticas:

* **Modularização:** O código é dividido em módulos lógicos (`data_loader.py`, `feature_engineering.py`, etc.) para melhor organização, legibilidade e reusabilidade.
* **Docstrings e Type Hints:** Todas as funções e métodos são documentados com `docstrings` explicando seu propósito, argumentos e retornos, e utilizam `type hints` para maior clareza e validação de tipos.
* **Tratamento de Erros:** O módulo `logging` é utilizado extensivamente para registrar informações, avisos e erros, facilitando a depuração e o monitoramento.
* **Operações Vetorizadas:** Sempre que possível, são utilizadas operações vetorizadas do `NumPy` e `Pandas` para otimizar o desempenho dos cálculos estatísticos e de features.
* **Geração de Gráficos:** Todos os gráficos são gerados com `Matplotlib` ou `Seaborn` e salvos na pasta `figures/` com resolução mínima de 150 dpi, garantindo alta qualidade visual.

## Orientações Gerais (do Requisito Original)

* **Grupos:** Este trabalho é ideal para grupos de no máximo 3 alunos.
* **Entrega:** A entrega final deve ser realizada até o dia 10/Jul/2025.
* **Compartilhamento:** O código deverá ser disponibilizado e compartilhado no Google Colab ou plataforma similar.
---


## Scripts Auxiliares para Escolha de Variáveis

A pasta `choose_var_training/` contém scripts para análise exploratória e seleção de variáveis para o treino dos modelos:

- **escolher_variaveis_treino.py**: Gera automaticamente heatmaps de correlação (com `annot=True`) para cada arquivo `.csv` em `data/processed`, excluindo a coluna `date`. Os gráficos são salvos como `heatmap_correlacao_<nome_do_arquivo>.png`.
- **otimizando_variaveis.py**: Executa regressão linear múltipla (usando `statsmodels`) para cada arquivo `.csv` em `data/processed`, excluindo a coluna `date`, e imprime o resumo estatístico do modelo. O script pode ser facilmente adaptado para testar diferentes combinações de variáveis. Nos comentários foram colocados os resultados encontrados

Esses scripts auxiliam na análise de multicolinearidade, importância e seleção das melhores features para os modelos de previsão.


Desenvolvido por: Thales Augusto Salvador, Carlos Henrique, Miguel Toledo(2025)