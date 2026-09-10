import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. UNIÃO DOS ARQUIVOS CSV
# Procura todos os arquivos .csv dentro das subpastas do projeto
caminho_arquivos = "./**/*.csv"
arquivos = glob.glob(caminho_arquivos, recursive=True)

print(f"Arquivos encontrados: {arquivos}")

# Lê e concatena todos os CSVs encontrados em um único DataFrame
df_lista = [pd.read_csv(f) for f in arquivos]
df = pd.concat(df_lista, ignore_index=True)


# 2. LIMPEZA DOS DADOS
# Remove linhas duplicadas
df.drop_duplicates(inplace=True)

# Remove linhas com valores ausentes (se houver)
df.dropna(inplace=True)


# 3. DIVISÃO ENTRE TREINO E TESTE
# Separa as features (X) da coluna alvo (y) -> substitua 'target' pelo nome correto da sua coluna
X = df.drop(columns=["target"])
y = df["target"]

# Separa 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. NORMALIZAÇÃO (Escalonamento das Features)
scaler = StandardScaler()

# Ajusta e transforma no Treino; apenas transforma no Teste
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converte de volta para DataFrame para manter os nomes das colunas organizados
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print(
    f"Pré-processamento concluído! Treino: {X_train_scaled.shape}, Teste: {X_test_scaled.shape}"
)