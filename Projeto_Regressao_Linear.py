# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Função para converter valores no formato "k", "m", "b" para números
def convert_to_number(value):
    if isinstance(value, str):
        if 'k' in value:
            return float(value.replace('k', '').replace(',', '')) * 1e3
        elif 'm' in value:
            return float(value.replace('m', '').replace(',', '')) * 1e6
        elif 'b' in value:
            return float(value.replace('b', '').replace(',', '')) * 1e9
        elif '%' in value:
            return float(value.replace('%', '')) / 100  # Converter porcentagem para decimal
    return float(value)

# Carregar os dados
df = pd.read_csv('top_insta_influencers_data.csv')

# Limpeza e conversão de colunas
columns_to_convert = ['posts', 'followers', 'avg_likes', '60_day_eng_rate', 
                      'new_post_avg_like', 'total_likes']
for col in columns_to_convert:
    df[col] = df[col].apply(convert_to_number)

# Análise exploratória - Correlação
plt.figure(figsize=(10, 6))
correlation_matrix = df[['followers', 'avg_likes', '60_day_eng_rate', 
                         'new_post_avg_like', 'posts']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa de Correlação Entre as Variáveis")
plt.show()

# Scatterplot - Relação entre seguidores e taxa de engajamento
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='followers', y='60_day_eng_rate', alpha=0.7)
plt.xscale('log')
plt.title("Relação entre Seguidores e Taxa de Engajamento")
plt.xlabel("Seguidores (log)")
plt.ylabel("Taxa de Engajamento")
plt.show()

# Remover valores ausentes da variável alvo e das variáveis preditoras
features = ['followers', 'avg_likes', 'new_post_avg_like']
target = '60_day_eng_rate'
df_cleaned = df.dropna(subset=features + [target])

# Separação de variáveis independentes e dependente
X = df_cleaned[features]
y = df_cleaned[target]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar e treinar o modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio (MSE): {mse}")
print(f"Erro Absoluto Médio (MAE): {mae}")
print(f"Coeficiente de Determinação (R²): {r2}")

# Garantindo limpeza completa dos resíduos
residuos = y_test - y_pred
residuos = np.nan_to_num(residuos, nan=0.0, posinf=0.0, neginf=0.0)

# Visualizações
# Gráfico: Valores Reais vs Previstos (com métricas)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, label='Previsões')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Linha de Identidade')  # Linha de identidade
plt.title("Valores Reais vs Previstos", fontsize=14)
plt.xlabel("Taxa de Engajamento Real", fontsize=12)
plt.ylabel("Taxa de Engajamento Prevista", fontsize=12)
plt.legend()
# Adicionando texto com métricas
plt.text(
    min(y_test), max(y_pred),
    f"$R^2$: {r2:.2f}\nMAE: {mae:.4f}\nMSE: {mse:.6f}",
    fontsize=10, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5")
)
plt.grid(alpha=0.3)
plt.show()

# Gráfico: Resíduos (com densidade simplificada)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuos, alpha=0.7, label='Resíduos')
plt.axhline(0, color='red', linestyle='--', label='Resíduo Zero')
plt.title("Análise de Resíduos", fontsize=14)
plt.xlabel("Taxa de Engajamento Prevista", fontsize=12)
plt.ylabel("Resíduos", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Coeficientes do modelo
coefficients = pd.DataFrame({
    "Variável": features,
    "Coeficiente": model.coef_
})
intercept = model.intercept_

print("Coeficientes do Modelo:")
print(coefficients)
print(f"Intercepto: {intercept}")
