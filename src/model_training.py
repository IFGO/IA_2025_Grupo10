import pandas as pd
import numpy as np
import logging
import os
import joblib # Para salvar/carregar modelos
import matplotlib
matplotlib.use('Agg') # <-- Adicionado: Define o backend não-interativo antes de importar pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    kfolds: int,
    pair_name: str,
    models_folder: str,
    poly_degree: int = 2
):
    """
    Treina e avalia um modelo de regressão usando K-fold cross-validation.
    Salva o modelo treinado e exibe as métricas de avaliação.

    Args:
        X (pd.DataFrame): Features de entrada.
        y (pd.Series): Variável alvo (preço de fechamento).
        model_type (str): Tipo de modelo ('MLP', 'Linear', 'Polynomial', 'RandomForest').
        kfolds (int): Número de folds para K-fold cross-validation.
        pair_name (str): Nome do par de criptomoedas para nomear arquivos.
        models_folder (str): Pasta para salvar os modelos treinados.
        poly_degree (int): Grau para a regressão polinomial, se aplicável.
    """
    logging.info(f"Iniciando treinamento e avaliação do modelo {model_type} para {pair_name}...")

    # Define o modelo com base no tipo
    model = None
    if model_type == 'MLP':
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=50)
    elif model_type == 'Linear':
        model = LinearRegression()
    elif model_type == 'Polynomial':
        if not (2 <= poly_degree <= 10):
            logging.error(f"Grau polinomial inválido: {poly_degree}. Deve estar entre 2 e 10.")
            return
        model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
    elif model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        logging.error(f"Tipo de modelo '{model_type}' não suportado.")
        return

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

    mse_scores = []
    mae_scores = []
    r2_scores = []

    # Certifica-se de que X e y têm os mesmos índices para KFold funcionar corretamente
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    # Verifica se há dados suficientes para o K-fold
    if len(X_reset) < kfolds:
        logging.warning(f"Não há dados suficientes para {kfolds}-Fold CV para {pair_name}. Mínimo de amostras: {kfolds}.")
        # Treina e salva o modelo no conjunto completo se não houver dados suficientes para CV
        if len(X_reset) > 0:
            try:
                model.fit(X_reset, y_reset)
                model_filename = os.path.join(models_folder, f"{model_type.lower()}_{pair_name.replace(' ', '_')}.pkl")
                joblib.dump(model, model_filename)
                logging.info(f"Modelo {model_type} para {pair_name} salvo (treinado no conjunto completo).")
            except Exception as e:
                logging.error(f"Erro ao salvar o modelo {model_type} para {pair_name}: {e}")
        return


    for fold, (train_index, test_index) in enumerate(kf.split(X_reset)):
        X_train, X_test = X_reset.iloc[train_index], X_reset.iloc[test_index]
        y_train, y_test = y_reset.iloc[train_index], y_reset.iloc[test_index]

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            logging.info(f"  Fold {fold+1}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        except Exception as e:
            logging.error(f"Erro durante o treinamento/avaliação do Fold {fold+1} para {model_type}: {e}")
            continue

    if mse_scores:
        avg_mse = np.mean(mse_scores)
        avg_mae = np.mean(mae_scores)
        avg_r2 = np.mean(r2_scores)

        logging.info(f"Resultados Médios para {model_type} em {pair_name} ({kfolds}-Fold CV):")
        logging.info(f"  MSE Médio: {avg_mse:.4f}")
        logging.info(f"  MAE Médio: {avg_mae:.4f}")
        logging.info(f"  R2 Médio: {avg_r2:.4f}")

        # Salva o modelo treinado (treina no conjunto completo para salvar)
        try:
            model.fit(X_reset, y_reset) # Treina no conjunto completo para uso futuro
            model_filename = os.path.join(models_folder, f"{model_type.lower()}_{pair_name.replace(' ', '_')}.pkl")
            joblib.dump(model, model_filename)
            logging.info(f"Modelo {model_type} para {pair_name} salvo em: {model_filename}")
        except Exception as e:
            logging.error(f"Erro ao salvar o modelo {model_type} para {pair_name}: {e}")
    else:
        logging.warning(f"Nenhum score foi calculado para {model_type} em {pair_name}.")


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    kfolds: int,
    pair_name: str,
    plots_folder: str,
    poly_degree: int = 2
):
    """
    Compara o desempenho de diferentes modelos de regressão e plota os resultados.

    Args:
        X (pd.DataFrame): Features de entrada.
        y (pd.Series): Variável alvo (preço de fechamento).
        kfolds (int): Número de folds para K-fold cross-validation.
        pair_name (str): Nome do par de criptomoedas para nomear arquivos.
        plots_folder (str): Pasta para salvar os gráficos de comparação.
        poly_degree (int): Grau para a regressão polinomial.
    """
    logging.info(f"Comparando modelos para {pair_name}...")

    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=50),
        'Linear Regression': LinearRegression(),
        'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression()),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = [] # Para armazenar MSE, MAE, R2 para cada modelo e fold

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    # Verifica se há dados suficientes para o K-fold
    if len(X_reset) < kfolds:
        logging.warning(f"Não há dados suficientes para {kfolds}-Fold CV para comparação de modelos em {pair_name}.")
        return

    for model_name, model in models.items():
        mse_scores_model = []
        mae_scores_model = []
        r2_scores_model = []
        std_error_model = [] # Para erro padrão

        for fold, (train_index, test_index) in enumerate(kf.split(X_reset)):
            X_train, X_test = X_reset.iloc[train_index], X_reset.iloc[test_index]
            y_train, y_test = y_reset.iloc[train_index], y_reset.iloc[test_index]

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                std_err = np.std(y_test - y_pred) # Erro padrão dos resíduos

                mse_scores_model.append(mse)
                mae_scores_model.append(mae)
                r2_scores_model.append(r2)
                std_error_model.append(std_err)

            except Exception as e:
                logging.error(f"Erro durante a comparação do modelo {model_name} no Fold {fold+1}: {e}")
                continue

        if mse_scores_model:
            results.append({
                'Model': model_name,
                'Avg MSE': np.mean(mse_scores_model),
                'Avg MAE': np.mean(mae_scores_model),
                'Avg R2': np.mean(r2_scores_model),
                'Avg Std Error': np.mean(std_error_model)
            })
        else:
            logging.warning(f"Nenhum score foi calculado para o modelo {model_name} em {pair_name}.")

    df_results = pd.DataFrame(results)
    print(f"\n*** Comparação de Modelos para {pair_name} ***\n")
    print(df_results.to_string())

    if not df_results.empty:
        # Identificar o melhor regressor (menor MSE ou MAE, maior R2)
        best_regressor = df_results.loc[df_results['Avg MSE'].idxmin()]
        logging.info(f"Melhor Regressor para {pair_name} (baseado em MSE): {best_regressor['Model']}")

        # Cálculo do erro padrão entre o MLP e o melhor regressor (Requisito 9e)
        mlp_row = df_results[df_results['Model'] == 'MLP']
        if not mlp_row.empty and best_regressor['Model'] != 'MLP':
            std_err_mlp = mlp_row['Avg Std Error'].iloc[0]
            std_err_best = best_regressor['Avg Std Error']
            logging.info(f"Erro Padrão do MLP: {std_err_mlp:.4f}")
            logging.info(f"Erro Padrão do Melhor Regressor ({best_regressor['Model']}): {std_err_best:.4f}")
            logging.info(f"Diferença no Erro Padrão (MLP vs Melhor): {abs(std_err_mlp - std_err_best):.4f}")

    # Plotar Diagrama de Dispersão (Requisito 9a)
    logging.info(f"Gerando diagrama de dispersão para {pair_name}...")

    plt.figure(figsize=(12, 8))
    sns.set_palette("viridis")

    X_full = X.reset_index(drop=True)
    y_full = y.reset_index(drop=True)

    for model_name, model in models.items():
        try:
            # Treina o modelo no conjunto completo para gerar previsões para o plot
            model.fit(X_full, y_full)
            y_pred_full = model.predict(X_full)
            plt.scatter(y_full, y_pred_full, alpha=0.6, label=f'{model_name} (R2: {r2_score(y_full, y_pred_full):.2f})')
        except Exception as e:
            logging.error(f"Erro ao gerar previsões para o diagrama de dispersão do modelo {model_name}: {e}")
            continue

    plt.plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 'k--', lw=2, label='Linha Ideal (Previsão = Real)')
    plt.xlabel('Preço Real (USDT)')
    plt.ylabel('Preço Previsto (USDT)')
    plt.title(f'Diagrama de Dispersão: Preço Real vs. Previsto para {pair_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_folder, f"scatter_plot_models_{pair_name.replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"Diagrama de dispersão salvo em: {plot_path}")

    # Coeficientes de Correlação dos Regressores (Requisito 9b) - para modelos lineares
    logging.info("Coeficientes dos Regressores (para modelos lineares/polinomiais):")
    for model_name, model in models.items():
        if 'Linear' in model_name: # Inclui Linear e Polynomial
            try:
                # Treina o modelo no conjunto completo para obter os coeficientes
                model.fit(X_full, y_full)
                if isinstance(model, make_pipeline): # Para Polynomial Regression
                    linear_model = model.named_steps['linearregression']
                    feature_names = model.named_steps['polynomialfeatures'].get_feature_names_out(X.columns)
                else: # Para Linear Regression
                    linear_model = model
                    feature_names = X.columns

                if hasattr(linear_model, 'coef_'):
                    logging.info(f"  Modelo: {model_name}")
                    for feature, coef in zip(feature_names, linear_model.coef_):
                        logging.info(f"    {feature}: {coef:.4f}")
            except Exception as e:
                logging.warning(f"Não foi possível obter coeficientes para {model_name}: {e}")

    # Determinar a equação que melhor representa os regressores (Requisito 9c)
    logging.info("Equações que melhor representam os regressores:")
    for model_name, model in models.items():
        if 'Linear' in model_name:
            try:
                model.fit(X_full, y_full) # Treina novamente para garantir coeficientes
                if isinstance(model, make_pipeline):
                    linear_model = model.named_steps['linearregression']
                    poly_features = model.named_steps['polynomialfeatures']
                    feature_names = poly_features.get_feature_names_out(X.columns)
                    intercept = linear_model.intercept_
                    equation = f"y = {intercept:.4f}"
                    for feature, coef in zip(feature_names, linear_model.coef_):
                        equation += f" + ({coef:.4f} * {feature})"
                    logging.info(f"  Modelo: {model_name} -> {equation}")
                else:
                    intercept = model.intercept_
                    equation = f"y = {intercept:.4f}"
                    for feature, coef in zip(X.columns, model.coef_):
                        equation += f" + ({coef:.4f} * {feature})"
                    logging.info(f"  Modelo: {model_name} -> {equation}")
            except Exception as e:
                logging.warning(f"Não foi possível determinar a equação para {model_name}: {e}")
        else:
            logging.info(f"  Modelo: {model_name} -> Não é uma equação linear simples.")

