import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Caminho para os arquivos processados
processed_folder = (
    os.path.join("..", "data", "processed")
    if not os.path.exists("data/processed")
    else "data/processed"
)

csv_files = glob.glob(os.path.join(processed_folder, "*.csv"))

for csv_path in csv_files:
    df = pd.read_csv(csv_path)  # type: ignore
    # Remove a coluna 'date' se existir
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    # Calcula a matriz de correlação
    corr = df.corr()
    # Nome do arquivo para o gráfico
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    plt.figure(figsize=(14, 10))  # type: ignore
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)  # type: ignore
    plt.title(f"Heatmap de Correlação - {base_name}")  # type: ignore
    plt.tight_layout()
    plt.savefig(f"heatmap_correlacao_{base_name}.png", dpi=150)  # type: ignore
    plt.close()
    print(f"Heatmap salvo: heatmap_correlacao_{base_name}.png")
