

# Analisi e Ottimizzazione del Consumo Energetico Residenziale

Questo progetto ha l'obiettivo di analizzare e ottimizzare il consumo di energia nelle abitazioni, utilizzando dati storici di consumo per ambiente. Le analisi mirano a identificare schemi, prevedere il consumo e proporre raccomandazioni per la riduzione dei costi e l’aumento della sostenibilità.

---

## Descrizione dei Dati

Il dataset (`df`) utilizzato contiene registrazioni giornaliere del consumo energetico (in kWh) in diversi ambienti residenziali su un determinato periodo. Le principali colonne del dataframe sono:

- **Data:** Data della misurazione.
- **Camera1, Camera2:** Consumo nelle camere 1 e 2.
- **Soggiorno:** Consumo nel soggiorno.
- **Cucina:** Consumo in cucina.
- **Piscina:** Consumo relativo all’uso della piscina.
- **kWh:** Consumo totale giornaliero dell’abitazione (kilowattora).

---

## Grafici e Analisi

### 1. Istogramma del Consumo Totale

```python
plt.figure(figsize=(8, 5))
plt.hist(df['kWh'], bins=20, color='skyblue')
plt.title('Distribuzione - Consumo Totale (kWh)')
plt.xlabel('Consumo Totale (kWh)')
plt.ylabel('Conteggio')
plt.show()
```

**Analisi:**
L’istogramma mostra che la maggior parte dei giorni presenta un consumo totale tra 1000 e 1300 kWh, con pochi giorni di consumo molto basso o molto alto. Questo indica un pattern relativamente stabile, ma con margini di ottimizzazione agli estremi.

---

### 2. Istogrammi di Tutte le Variabili

```python
import seaborn as sns

cols_to_plot = df.columns[1:]  # Ignora la colonna 'Data'
n_cols = len(cols_to_plot)
n_rows = (n_cols + 2) // 3  # 3 colonne per riga

fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=10)
    axes[i].set_title(f'Distribuzione - {col}')
    axes[i].set_xlabel(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Distribuzione delle Variabili di Consumo", fontsize=16, y=1.02)
plt.show()
```

**Analisi:**
Gli istogrammi individuali rivelano che soggiorno e piscina mostrano una maggiore variazione d’uso, mentre camere e cucina hanno distribuzioni più concentrate. Questo suggerisce che interventi in questi ambienti possono avere un impatto maggiore nella riduzione del consumo.

---

### 3. Mappa di Correlazione

```python
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=['Data']).corr(), annot=True, cmap='Blues', fmt=".2f")
plt.title('Mappa di Correlazione tra Variabili')
plt.show()
```

**Analisi:**
L’analisi di correlazione mostra che il soggiorno è l’ambiente più associato all’aumento del consumo totale (correlazione 0.55), seguito da Camera1 (0.52) e Piscina (0.43). Cucina e Camera2 influenzano anch’essi, ma con peso minore. La cucina ha una correlazione negativa con la piscina (-0.08), suggerendo un uso alternato in alcuni giorni.

---

### 4. Consumo Totale nel Tempo

```python
plt.figure(figsize=(14, 6))
plt.plot(df['Data'], df['kWh'], label='Consumo Totale (kWh)', color='blue', linewidth=2)
plt.title('Evoluzione del Consumo Totale di Energia')
plt.xlabel('Data')
plt.ylabel('kWh')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

**Analisi:**
Il grafico temporale permette di identificare picchi e stagionalità nei consumi, fondamentali per interventi programmati e campagne educative.

---

### 5. Consumo per Ambiente nel Tempo

```python
plt.figure(figsize=(16, 8))
for ambiente in ['Camera1', 'Camera2', 'Soggiorno', 'Cucina', 'Piscina']:
    plt.plot(df['Data'], df[ambiente], label=ambiente)

plt.title('Consumo Energetico per Ambiente nel Tempo')
plt.xlabel('Data')
plt.ylabel('Consumo (kWh)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

**Analisi:**
Visualizzando il consumo per ambiente nel tempo, è possibile identificare quali ambienti contribuiscono maggiormente ai picchi di consumo e indirizzare azioni specifiche.

---

### 6. Regressione Lineare: Reale vs Previsto

```python
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Linea identità
plt.xlabel('Consumo Reale (kWh)')
plt.ylabel('Consumo Previsto (kWh)')
plt.title('Reale vs Previsto - Consumo Totale')
plt.show()
```

**Dettagli del Modello:**

- **Variabili predittive:** Camera1, Camera2, Soggiorno, Cucina, Piscina.
- **Variabile target:** Consumo totale (kWh).
- **Validazione:** Holdout (train/test).
- **Metriche:** R² = 0.86, RMSE = XX, MAE = XX (sostituire XX con i valori reali, se disponibili).

**Analisi:**
Il modello di regressione lineare ha mostrato un R² di 0.86, indicando un’elevata capacità predittiva. Gli ambienti che più impattano il consumo totale sono soggiorno, piscina e cucina.

---

### 7. Metodo del Gomito (KMeans)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = df[['Camera1', 'Camera2', 'Soggiorno', 'Cucina', 'Piscina']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title('Metodo del Gomito')
plt.xlabel('Numero di cluster')
plt.ylabel('Inerzia (somma delle distanze quadratiche)')
plt.show()
```

> **Nota:** L’inerzia rappresenta la somma delle distanze quadratiche dei punti dai rispettivi centroidi. Il "gomito" indica il numero ideale di cluster.

**Analisi:**
Il metodo del gomito suggerisce che 3 cluster siano ideali per segmentare i profili di consumo.

---

### 8. Visualizzazione dei Cluster

```python
import seaborn as sns

sns.pairplot(df, hue='Cluster', vars=['Camera1', 'Camera2', 'Soggiorno', 'Cucina', 'Piscina'], palette='tab10')
plt.suptitle("Pattern di Uso Raggruppati per Cluster", y=1.02)
plt.show()
```

**Analisi:**
La clusterizzazione ha identificato tre gruppi principali:

- **Basso consumo:** Uso ridotto in tutti gli ambienti.
- **Consumo equilibrato:** Uso moderato e distribuito.
- **Alto consumo:** Picchi in soggiorno, cucina o piscina.

Questi gruppi permettono di personalizzare raccomandazioni e interventi per ciascun profilo.

---

## Profili e Raccomandazioni

- **Basso Consumo:** Mantenere le buone pratiche, incentivare l’energia solare e il monitoraggio continuo.
- **Consumo Equilibrato:** Automatizzare lo spegnimento degli apparecchi, installare sensori di presenza e timer.
- **Alto Consumo:** Automatizzare le luci, monitorare l’uso della piscina, incentivare l’uso consapevole e considerare apparecchi più efficienti.

---

## Conclusione

I dati hanno permesso di segmentare le abitazioni e prevedere il consumo con alta precisione. Il modello è semplice, interpretabile e indica quali ambienti influenzano maggiormente il consumo totale. L’analisi consente raccomandazioni mirate e interventi per la riduzione dei costi e l’aumento della sostenibilità.

> **Prossimi passi:**
> - Integrare dati in tempo reale e IoT per un monitoraggio dinamico.
> - Esplorare modelli predittivi più avanzati.
> - Valutare l’impatto delle raccomandazioni implementate.

---

**Domande, suggerimenti o interesse a collaborare? Contattaci!**



