import pandas as pd
import numpy as np
import logging
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
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
    Treina, avalia e salva um modelo de regressão usando K-fold cross-validation.

    Args:
        X (pd.DataFrame): DataFrame com as features de entrada.
        y (pd.Series): Series com a variável alvo.
        model_type (str): O tipo de modelo a ser treinado ('MLP', 'Linear',
                          'Polynomial', 'RandomForest').
        kfolds (int): O número de folds para a validação cruzada.
        pair_name (str): O nome do par de moedas para nomear os ficheiros.
        models_folder (str): O diretório para salvar o modelo treinado.
        poly_degree (int): O grau a ser usado na regressão polinomial.
    """
    logging.info(f"Iniciando treino e avaliação do modelo {model_type} para {pair_name}...")

    model_mapping = {
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=50),
        'Linear': LinearRegression(),
        'Polynomial': make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression()),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    if model_type not in model_mapping:
        logging.error(f"Tipo de modelo '{model_type}' não suportado.")
        return

    if model_type == 'Polynomial' and not (2 <= poly_degree <= 10):
        logging.error(f"Grau polinomial inválido: {poly_degree}. Deve estar entre 2 e 10.")
        return

    model = model_mapping[model_type]
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    X_reset, y_reset = X.reset_index(drop=True), y.reset_index(drop=True)

    if len(X_reset) < kfolds:
        logging.warning(f"Dados insuficientes para {kfolds}-Fold CV em {pair_name}. Treinando no conjunto completo.")
        if len(X_reset) > 0:
            try:
                model.fit(X_reset, y_reset)
                model_filename = os.path.join(models_folder, f"{model_type.lower()}_{pair_name.replace(' ', '_')}.pkl")
                joblib.dump(model, model_filename)
                logging.info(f"Modelo {model_type} para {pair_name} salvo em: {model_filename}")
            except Exception as e:
                logging.error(f"Erro ao salvar modelo {model_type} para {pair_name}: {e}")
        return

    mse_scores = np.zeros(kfolds)
    mae_scores = np.zeros(kfolds)
    r2_scores = np.zeros(kfolds)

    for i, (train_index, test_index) in enumerate(kf.split(X_reset)):
        X_train, X_test = X_reset.iloc[train_index], X_reset.iloc[test_index]
        y_train, y_test = y_reset.iloc[train_index], y_reset.iloc[test_index]

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores[i] = mean_squared_error(y_test, y_pred)
            mae_scores[i] = mean_absolute_error(y_test, y_pred)
            r2_scores[i] = r2_score(y_test, y_pred)
            logging.info(f"  Fold {i+1}: MSE={mse_scores[i]:.4f}, MAE={mae_scores[i]:.4f}, R2={r2_scores[i]:.4f}")
        except Exception as e:
            logging.error(f"Erro no Fold {i+1} para {model_type}: {e}")
            mse_scores[i], mae_scores[i], r2_scores[i] = np.nan, np.nan, np.nan

    if not np.isnan(mse_scores).all():
        logging.info(f"Resultados Médios para {model_type} ({kfolds}-Fold CV):")
        logging.info(f"  MSE Médio: {np.nanmean(mse_scores):.4f}")
        logging.info(f"  MAE Médio: {np.nanmean(mae_scores):.4f}")
        logging.info(f"  R2 Médio: {np.nanmean(r2_scores):.4f}")

        try:
            model.fit(X_reset, y_reset)
            model_filename = os.path.join(models_folder, f"{model_type.lower()}_{pair_name.replace(' ', '_')}.pkl")
            joblib.dump(model, model_filename)
            logging.info(f"Modelo final {model_type} salvo em: {model_filename}")
        except Exception as e:
            logging.error(f"Erro ao salvar modelo final {model_type}: {e}")
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
    Compara múltiplos modelos de regressão, exibe métricas e gera gráficos.

    Args:
        X (pd.DataFrame): DataFrame com as features de entrada.
        y (pd.Series): Series com a variável alvo.
        kfolds (int): O número de folds para a validação cruzada.
        pair_name (str): O nome do par de moedas para nomear os ficheiros.
        plots_folder (str): O diretório para salvar os gráficos de comparação.
        poly_degree (int): O grau a ser usado na regressão polinomial.
    """
    logging.info(f"Comparando modelos para {pair_name}...")

    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=50),
        'Linear Regression': LinearRegression(),
        'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression()),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    X_reset, y_reset = X.reset_index(drop=True), y.reset_index(drop=True)

    if len(X_reset) < kfolds:
        logging.warning(f"Dados insuficientes para {kfolds}-Fold CV para comparação em {pair_name}.")
        return

    for model_name, model in models.items():
        mse_scores = np.zeros(kfolds)
        mae_scores = np.zeros(kfolds)
        r2_scores = np.zeros(kfolds)
        std_error_scores = np.zeros(kfolds)

        for i, (train_index, test_index) in enumerate(kf.split(X_reset)):
            X_train, X_test = X_reset.iloc[train_index], X_reset.iloc[test_index]
            y_train, y_test = y_reset.iloc[train_index], y_reset.iloc[test_index]

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse_scores[i] = mean_squared_error(y_test, y_pred)
                mae_scores[i] = mean_absolute_error(y_test, y_pred)
                r2_scores[i] = r2_score(y_test, y_pred)
                std_error_scores[i] = np.std(y_test - y_pred)
            except Exception as e:
                logging.error(f"Erro na comparação do modelo {model_name} no Fold {i+1}: {e}")
                mse_scores[i], mae_scores[i], r2_scores[i], std_error_scores[i] = np.nan, np.nan, np.nan, np.nan
        
        if not np.isnan(mse_scores).all():
            results.append({
                'Model': model_name,
                'Avg MSE': np.nanmean(mse_scores),
                'Avg MAE': np.nanmean(mae_scores),
                'Avg R2': np.nanmean(r2_scores),
                'Avg Std Error': np.nanmean(std_error_scores)
            })

    if not results:
        logging.warning(f"Nenhum resultado de modelo foi gerado para {pair_name}.")
        return

    df_results = pd.DataFrame(results)
    logging.info(f"\n*** Comparação de Modelos para {pair_name} ***\n{df_results.to_string()}")

    best_regressor = df_results.loc[df_results['Avg MSE'].idxmin()]
    logging.info(f"Melhor Regressor para {pair_name} (baseado em MSE): {best_regressor['Model']}")

    # Cálculo do erro padrão entre o MLP e o melhor regressor
    mlp_row = df_results[df_results['Model'] == 'MLP']
    if not mlp_row.empty and best_regressor['Model'] != 'MLP':
        std_err_mlp = mlp_row['Avg Std Error'].iloc[0]
        std_err_best = best_regressor['Avg Std Error']
        logging.info(f"Erro Padrão do MLP: {std_err_mlp:.4f}")
        logging.info(f"Erro Padrão do Melhor Regressor ({best_regressor['Model']}): {std_err_best:.4f}")
        logging.info(f"Diferença no Erro Padrão (MLP vs Melhor): {abs(std_err_mlp - std_err_best):.4f}")

    _plot_scatter_comparison(X_reset, y_reset, models, pair_name, plots_folder)
    _log_coefficients(X_reset, y_reset, models, pair_name)


def _plot_scatter_comparison(X, y, models, pair_name, plots_folder):
    """Função auxiliar para plotar o diagrama de dispersão de Real vs. Previsto."""
    plt.figure(figsize=(12, 8))
    sns.set_palette("viridis")
    for model_name, model in models.items():
        try:
            model.fit(X, y)
            y_pred = model.predict(X)
            plt.scatter(y, y_pred, alpha=0.6, label=f'{model_name} (R2: {r2_score(y, y_pred):.2f})')
        except Exception as e:
            logging.error(f"Erro ao gerar previsões para dispersão do modelo {model_name}: {e}")

    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Linha Ideal')
    plt.xlabel('Preço Real (USDT)')
    plt.ylabel('Preço Previsto (USDT)')
    plt.title(f'Diagrama de Dispersão: Real vs. Previsto para {pair_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_folder, f"scatter_plot_models_{pair_name.replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"Diagrama de dispersão salvo em: {plot_path}")


def _log_coefficients(X, y, models, pair_name):
    """Função auxiliar para logar os coeficientes de modelos lineares."""
    logging.info(f"Análise de Coeficientes e Equações para {pair_name}:")
    for model_name, model in models.items():
        if 'Linear' not in model_name and 'Polynomial' not in model_name:
            logging.info(f"  Modelo: {model_name} -> Não é uma equação linear simples.")
            continue
        try:
            model.fit(X, y)
            if isinstance(model, Pipeline):
                linear_model = model.named_steps['linearregression']
                poly_features = model.named_steps['polynomialfeatures']
                feature_names = poly_features.get_feature_names_out(X.columns)
                intercept = linear_model.intercept_
                coefs = linear_model.coef_
            else:
                intercept = model.intercept_
                coefs = model.coef_
                feature_names = X.columns

            equation = f"y = {intercept:.4f}"
            for feature, coef in zip(feature_names, coefs):
                equation += f" + ({coef:.4f} * {feature})"
            logging.info(f"  Modelo: {model_name} -> {equation}")
        except Exception as e:
            logging.warning(f"Não foi possível determinar a equação para {model_name}: {e}")

