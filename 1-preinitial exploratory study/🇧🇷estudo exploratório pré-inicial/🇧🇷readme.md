

# Análise e Otimização do Consumo de Energia Residencial

Este projeto tem como objetivo analisar e otimizar o consumo de energia em residências, utilizando dados históricos de consumo por cômodo. As análises visam identificar padrões, prever consumo e propor recomendações para redução de custos e aumento da sustentabilidade.

---

## Descrição dos Dados

O conjunto de dados (`df`) utilizado contém registros diários de consumo de energia (em kWh) em diferentes ambientes residenciais ao longo de um determinado período. As principais colunas do dataframe são:

- **Data:** Data da medição.
- **Quarto1, Quarto2:** Consumo nos quartos 1 e 2.
- **Sala:** Consumo na sala de estar.
- **Cozinha:** Consumo na cozinha.
- **Piscina:** Consumo relacionado ao uso da piscina.
- **kWh:** Consumo total diário da residência (quilowatt-hora).

---

## Plots e Análises

### 1. Histograma do Consumo Total

```python
plt.figure(figsize=(8, 5))
plt.hist(df['kWh'], bins=20, color='skyblue')
plt.title('Distribuição - Consumo Total (kWh)')
plt.xlabel('Consumo Total (kWh)')
plt.ylabel('Contagem')
plt.show()
```

**Análise:**
O histograma mostra que a maioria dos dias apresenta consumo total entre 1000 e 1300 kWh, com poucos dias de consumo muito baixo ou muito alto. Isso indica um padrão relativamente estável, mas com espaço para otimização nos extremos.

---

### 2. Histogramas de Todas as Variáveis

```python
import seaborn as sns

cols_to_plot = df.columns[1:]  # Ignora a coluna 'Data'
n_cols = len(cols_to_plot)
n_rows = (n_cols + 2) // 3  # 3 colunas por linha

fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=10)
    axes[i].set_title(f'Distribuição - {col}')
    axes[i].set_xlabel(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Distribuição das Variáveis de Consumo", fontsize=16, y=1.02)
plt.show()
```

**Análise:**
Os histogramas individuais revelam que sala e piscina apresentam maior variação de uso, enquanto quartos e cozinha têm distribuições mais concentradas. Isso sugere que ações nesses ambientes podem ter maior impacto na redução do consumo.

---

### 3. Mapa de Correlação

```python
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=['Data']).corr(), annot=True, cmap='Blues', fmt=".2f")
plt.title('Mapa de Correlação entre Variáveis')
plt.show()
```

**Análise:**
A análise de correlação mostra que a sala é o cômodo mais associado ao aumento do consumo total (correlação 0.55), seguida por Quarto1 (0.52) e Piscina (0.43). Cozinha e Quarto2 também influenciam, mas com menor peso. A Cozinha tem correlação negativa com a Piscina (-0.08), sugerindo uso alternado em alguns dias.

---

### 4. Consumo Total ao Longo do Tempo

```python
plt.figure(figsize=(14, 6))
plt.plot(df['Data'], df['kWh'], label='Consumo Total (kWh)', color='blue', linewidth=2)
plt.title('Evolução do Consumo Total de Energia')
plt.xlabel('Data')
plt.ylabel('kWh')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

**Análise:**
O gráfico temporal permite identificar picos e sazonalidades no consumo, fundamentais para intervenções programadas e campanhas educativas.

---

### 5. Consumo por Cômodo ao Longo do Tempo

```python
plt.figure(figsize=(16, 8))
for comodo in ['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'Piscina']:
    plt.plot(df['Data'], df[comodo], label=comodo)

plt.title('Consumo de Energia por Cômodo ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Consumo (kWh)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

**Análise:**
Visualizando o consumo por cômodo ao longo do tempo, é possível identificar quais ambientes mais contribuem para os picos de consumo e direcionar ações específicas.

---

### 6. Regressão Linear: Real vs Previsto

```python
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Linha identidade
plt.xlabel('Consumo Real (kWh)')
plt.ylabel('Consumo Previsto (kWh)')
plt.title('Real vs Previsto - Consumo Total')
plt.show()
```

**Detalhes do Modelo:**

- **Variáveis preditoras:** Quarto1, Quarto2, Sala, Cozinha, Piscina.
- **Variável alvo:** Consumo total (kWh).
- **Validação:** Holdout (treino/teste).
- **Métricas:** R² = 0.86, RMSE = XX, MAE = XX (substitua XX pelos valores reais, se disponíveis).

**Análise:**
O modelo de regressão linear apresentou R² de 0.86, indicando alta capacidade preditiva. Os principais ambientes que impactam o consumo total são Sala, Piscina e Cozinha.

---

### 7. Método do Cotovelo (KMeans)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = df[['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'Piscina']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia (soma das distâncias quadráticas)')
plt.show()
```

> **Nota:** Inércia representa a soma das distâncias quadráticas dos pontos aos seus respectivos centróides. O "cotovelo" indica o número ideal de clusters.

**Análise:**
O método do cotovelo sugere que 3 clusters são ideais para segmentar os perfis de consumo.

---

### 8. Visualização dos Clusters

```python
import seaborn as sns

sns.pairplot(df, hue='Cluster', vars=['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'Piscina'], palette='tab10')
plt.suptitle("Padrões de Uso Agrupados por Cluster", y=1.02)
plt.show()
```

**Análise:**
A clusterização identificou três grupos principais:

- **Baixo consumo:** Uso reduzido em todos os ambientes.
- **Consumo equilibrado:** Uso moderado e distribuído.
- **Alto consumo:** Picos em sala, cozinha ou piscina.

Esses grupos permitem personalizar recomendações e intervenções para cada perfil.

---

## Perfis e Recomendações

- **Baixo Consumo:** Manter boas práticas, incentivar energia solar e monitoramento contínuo.
- **Consumo Equilibrado:** Automatizar desligamento de equipamentos, instalar sensores de presença e temporizadores.
- **Consumo Elevado:** Automatizar luzes, monitorar uso da piscina, incentivar uso consciente e considerar equipamentos mais eficientes.

---

## Conclusão

Os dados permitiram segmentar residências e prever consumo com alta precisão. O modelo é simples, interpretável e indica os cômodos que mais influenciam o consumo total. A análise possibilita recomendações direcionadas e intervenções para redução de custos e aumento da sustentabilidade.

> **Próximos passos:**
> - Integrar dados em tempo real e IoT para monitoramento dinâmico.
> - Explorar modelos preditivos mais avançados.
> - Avaliar impacto das recomendações implementadas.

---

**Dúvidas, sugestões ou interesse em colaborar? Entre em contato!**



