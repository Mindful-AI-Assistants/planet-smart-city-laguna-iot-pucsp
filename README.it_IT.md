<br>

 \[**[🇮🇹 Italiano](README.it_IT.md)**\]  \[[🇧🇷 Português](README.pt_BR.md)\] \[[🇺🇸 English](README.md)\]

<br><br><br><br>

## <p align="center"> 🌐 CDIA Nexus PUC-SP: Centro di Innovazione per Soluzioni Intelligenti di Acqua ed Energia — Smart City Laguna IoT, Fortaleza, Brasile 
#### <p align="center"> ***Progetto per il monitoraggio, la previsione e l’ottimizzazione del consumo energetico in una casa intelligente, utilizzando [IoT]() e [IA](). Sviluppato nel contesto della [Smart City Laguna]() – CDIA PUC-SP***.

  <br><br>
 

<p align="center">
   <a href="https://github.com/sponsors/Mindful-AI-Assistants">
    <img src="https://img.shields.io/badge/Sponsorizza-Mindful%20AI%20Assistants-brightgreen?logo=GitHub&style=flat-square">
  </a>
</p>


<br><br>


#### <p align="center"> ***In collaborazione con [Planet]() Smart City, [PUC-SP]() - Scienza dei Dati & IA, [ONU]() Obiettivi di Sviluppo Sostenibile (SDG), [Starlink]() e [Proptech]() Brasile***


 <br><br>

 
<!-- VIDEO -->

https://github.com/user-attachments/assets/bb4652e1-8b0e-455c-a2ae-ba81e39a9095


#### 📺 [Guarda in Full HD su YouTube](https://youtu.be/cgW4ql2XQgo?si=clGglP8HF_zz3xc8)


<br><br><br>


#### <p align="center"><em> Esplora il [Simulatore]() e supporta il progetto di IA per le Città Intelligenti </em></p>

<br>

### <p align="center"> [⇩]()💦



<br>

<p align="center">
  <a href="https://projetosmartcitylagunacdia03-yh6luzcrafamiabtp5xn8m.streamlit.app/">
    <img 
      src="https://img.shields.io/badge/Simulatore_Smart_City_Laguna-Streamlit-brightgreen?logo=streamlit&logoColor=white&style=flat-square" 
      alt="Simulatore Smart City Laguna"
      style="height: 30px; width: auto;"
  </a>
</p>


<br><br><br>
































////////------------------------

<br<br><br><br><br><br><br><br>

## 📓 Pipeline del Codice

<br>

### **Cella 1 — Importazione delle librerie**

```python
import locale
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
```


<br>

### **Cella 2 — Caricamento dei dati**

```python
# Cambia il percorso secondo il tuo ambiente
file_path = "/Users/fabicampanari/Desktop/Project Planet Smart City Laguna/2-CRISP-DM - Project Smart City Laguna/🇧🇷 CRISP-DM_Projeto_Smart_City_Laguna/Consumo_de_Energia_Analise.xlsx"
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
print(sheet_names)
df = xls.parse('Sheet1')
print(df.head())
df.info()
```


<br>

### **Cella 3 — Preprocessing delle date**

```python
mesi_it = {
    'gen': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'mag': '05', 'giu': '06', 'lug': '07',
}
df['Data'] = df['Data'].astype(str)
df['Data'] = df['Data'].str.lower().replace(mesi_it, regex=True)
df['Data'] = pd.to_datetime(df['Data'] + '/2025', format='%d/%m/%Y')
```


<br>

### **Cella 4 — Statistiche descrittive e correlazione**

```python
summary = df.describe()
correlation = df.corr(numeric_only=True)
print(summary)
print(correlation)
```


<br>

### **Cella 5 — [PLOT 1] Distribuzione delle variabili**

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
cols = df.columns[1:7]
for i, col in enumerate(cols):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=10)
    axes[i].set_title(f'Distribuzione - {col}')
    axes[i].set_xlabel(col)
plt.tight_layout()
plt.suptitle("Distribuzione delle Variabili", fontsize=16, y=1.02)
plt.show()
```

**Inserisci qui per visualizzare la distribuzione di consumi e attivazioni.**

<br>

### **Cella 6 — [PLOT 2] Consumo totale nel tempo**

```python
plt.figure(figsize=(14, 6))
plt.plot(df['Data'], df['KW/H'], label='Consumo Totale (KW/H)', color='blue', linewidth=2)
plt.title('Evoluzione del Consumo Totale di Energia')
plt.xlabel('Data')
plt.ylabel('KW/H')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

**Mostra l’evoluzione del consumo nel tempo.**

<br>

### **Cella 7 — Raggruppamento settimanale e [PLOT 3] Attivazioni settimanali per stanza**

```python
df['Settimana'] = df['Data'].dt.to_period('W').apply(lambda r: r.start_time)
df_settimana = df.groupby('Settimana')[['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'Piscina']].sum()
df_settimana.plot(figsize=(12, 6), marker='o')
plt.title('Attivazioni Settimanali per Stanza')
plt.ylabel('Numero di Attivazioni')
plt.xlabel('Settimana')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Mostra la somma delle attivazioni settimanali per stanza.**

<br>

### **Cella 8 — [PLOT 4] Correlazione tra attivazioni e consumo**

```python
correlazioni = df[['KW/H', 'Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'Piscina']].corr()['KW/H'][1:]
plt.figure(figsize=(10, 5))
sns.barplot(x=correlazioni.index, y=correlazioni.values, palette='Oranges_r')
plt.title('Correlazione tra Attivazioni e Consumo di Energia (kWh)')
plt.ylabel('Correlazione')
plt.xlabel('Stanza')
plt.tight_layout()
plt.show()
```

**Mostra quali stanze impattano maggiormente il consumo totale.**

<br>

### **Cella 9 — Modellazione predittiva (Regressione Lineare) e valutazione**

```python
X = df[['Quarto1', 'Quarto2', 'Sala', 'Cozinha']]
y = df['KW/H']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modello = LinearRegression()
modello.fit(X_train, y_train)
y_pred = modello.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Errore quadratico medio (MSE):", round(mse, 2))
print("R²:", round(r2, 2))
```


<br>

### **Cella 10 — [PLOT 5] Consumo reale vs previsto**

```python
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Consumo reale (kWh)")
plt.ylabel("Consumo previsto (kWh)")
plt.title("Consumo Reale vs Previsto")
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Valuta visivamente le prestazioni del modello predittivo.**

<br>

### **Cella 11 — Coefficienti del modello**

```python
coefficienti = pd.Series(modello.coef_, index=X.columns)
print("\nContributo di ogni stanza nella previsione (coefficienti):")
print(coefficienti.sort_values(ascending=False))
```

**Mostra il peso di ogni stanza nella previsione del consumo.**

<br>

### **Cella 12 — Calcolo percentuali di attivazione per stanza**

```python
df['Totale_attivazioni'] = df[['Quarto1', 'Quarto2', 'Sala', 'Cozinha']].sum(axis=1)
for stanza in ['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'Piscina']:
    df[f'{stanza}_pct'] = df[stanza] / df['Totale_attivazioni']
```


<br>

### **Cella 13 — [PLOT 6] Metodo del Gomito per KMeans**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Metodo del Gomito')
plt.xlabel('Numero di cluster')
plt.ylabel('Inerzia')
plt.grid(True)
plt.show()
```

**Scegli visivamente il numero ideale di cluster.**

<br>

### **Cella 14 — KMeans e [PLOT 7] Pairplot dei cluster**

```python
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
sns.pairplot(df, hue='Cluster', vars=['Quarto1', 'Quarto2', 'Sala', 'Cozinha'], palette='tab10')
plt.suptitle("Pattern di Utilizzo Raggruppati per Cluster", y=1.02)
plt.show()
```

**Visualizza i raggruppamenti dei profili di consumo.**

<br>

### **Cella 15 — Profilo medio per cluster e denominazione**

```python
col_pcts = [f'{c}_pct' for c in ['Quarto1', 'Quarto2', 'Sala', 'Cozinha']]
profilo_clusters = df.groupby('Cluster')[['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'KW/H'] + col_pcts].mean()

def denomina_cluster(row):
    media_kw = df['KW/H'].mean()
    if row['KW/H'] < media_kw * 0.75:
        consumo_totale = '🔵 Basso Consumo'
    elif row['KW/H'] > media_kw * 1.25:
        consumo_totale = '🔴 Alto Consumo'
    else:
        consumo_totale = '🟡 Consumo Bilanciato'
    alti = []
    for stanza in ['Quarto1', 'Quarto2', 'Sala', 'Cozinha']:
        media_pct = df[f'{stanza}_pct'].mean()
        if row[f'{stanza}_pct'] > media_pct * 1.2:
            alti.append(stanza)
    if consumo_totale == '🔵 Basso Consumo':
        return consumo_totale
    if consumo_totale == '🟡 Consumo Bilanciato':
        if len(alti) == 0:
            return consumo_totale
        else:
            return f"🟠 Consumo Elevato in {', '.join(alti)}"
    if consumo_totale == '🔴 Alto Consumo':
        if len(alti) == 0:
            return consumo_totale
        else:
            return f"🔴 Alto Consumo (In {', '.join(alti)})"

profilo_clusters['Profilo'] = profilo_clusters.apply(denomina_cluster, axis=1)
```


<br>

### **Cella 16 — Dizionario raccomandazioni e stampa per cluster**

```python
def mappa_profilo_chiave(profilo):
    if profilo == '🔵 Basso Consumo':
        return profilo
    if profilo == '🟡 Consumo Bilanciato':
        return profilo
    if profilo.startswith('🟠 Consumo Elevato'):
        return '🟠 Consumo Elevato'
    if profilo.startswith('🔴 Alto Consumo'):
        if 'In' in profilo:
            idx = profilo.index('In') + 3
            testo = profilo[idx:]
            principale = testo.split(',')[^0].strip()
            if principale in ['Sala']:
                return '🔴 Alto Consumo (Sala/Cozinha)'
            elif principale == 'Cozinha':
                return '🔴 Alto Consumo (Cozinha)'
            else:
                return '🔴 Alto Consumo'
        else:
            return '🔴 Alto Consumo'
    return profilo

raccomandazioni = {
    '🔵 Basso Consumo': [
        "✅ Mantieni le buone pratiche già adottate.",
        "🎁 Offri premi o sconti (gamification).",
        "🔋 Incentiva l'uso di energia solare/microgenerazione."
    ],
    '🟡 Consumo Bilanciato': [
        "🔌 Automatizza lo spegnimento degli apparecchi in orari fissi.",
        "🕵️ Installa sensori di presenza in camere e soggiorno.",
        "📊 Invia report settimanali di confronto dei consumi."
    ],
    '🟠 Consumo Elevato': [
        "🛏️ Automatizza luci ed elettronica nelle stanze a consumo elevato.",
        "🕵️ Installa sensori di presenza specifici.",
        "📊 Monitora l'uso per identificare picchi inutili."
    ],
    '🔴 Alto Consumo (Sala/Cozinha)': [
        "💧 Pianifica la pompa della cucina fuori dagli orari di punta.",
        "💡 Incentiva l'uso consapevole di luci ed elettronica.",
        "🧠 Suggerisci automazione e tariffa bianca."
    ],
    '🔴 Alto Consumo (Cozinha)': [
        "🍳 Controlla gli apparecchi della cucina per consumi eccessivi.",
        "⏰ Controlla gli orari di uso di forno e frigo.",
        "💡 Incentiva l'uso efficiente dell'illuminazione."
    ]
}

for cluster_id, row in profilo_clusters.iterrows():
    print(f"\n=== Cluster {cluster_id} - {row['Profilo']} ===")
    print("📊 Profilo medio di consumo (attivazioni e kWh):")
    print(row[['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'KW/H']])
    print("\n📈 Percentuale media di attivazioni per stanza (%):")
    print((row[col_pcts] * 100).round(2))
    print("\n💡 Raccomandazioni:")
    chiave = mappa_profilo_chiave(row['Profilo'])
    if chiave in raccomandazioni:
        for rec in raccomandazioni[chiave]:
            print("-", rec)
    else:
        print("- Nessuna raccomandazione specifica per questo profilo.")
```


<br>

### **Cella 17 — [PLOT 8] Boxplot consumo per cluster**

```python
plt.figure(figsize=(7,5))
sns.boxplot(x='Cluster', y='KW/H', data=df)
plt.title('Distribuzione Consumo (KW/H) per Cluster')
plt.show()
```

**Mostra la variazione del consumo per cluster.**

<br>

### **Cella 18 — [PLOT 9] Heatmap percentuali per cluster**

```python
heatmap_data = profilo_clusters[col_pcts] * 100
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Percentuale di Attivazioni per Stanza (%)')
plt.xlabel('Stanze')
plt.ylabel('Cluster')
plt.show()
```

**Visualizza la distribuzione delle attivazioni per cluster.**

<br>

### **Cella 19 — [PLOT 10] Radar delle stanze per cluster**

```python
categorie = ['Quarto1', 'Quarto2', 'Sala', 'Cozinha']
angles = np.linspace(0, 2 * np.pi, len(categorie), endpoint=False).tolist()
angles += angles[:1]
plt.figure(figsize=(10, 8))
for i, row in profilo_clusters.iterrows():
    valori = [row[cat] for cat in categorie]
    valori += valori[:1]
    plt.polar(angles, valori, label=f'Cluster {i}')
plt.xticks(angles[:-1], categorie)
plt.title('Radar delle Stanze per Cluster')
plt.legend()
plt.show()
```

**Confronta il profilo di attivazione di ciascun cluster.**


<br>

### **Cella 20 — [PLOT 11] Visualizzazione cluster con PCA**

<br>

Abbiamo applicato il PCA a scopo dimostrativo, anche con solo due cluster, per mostrare come agisce nella riduzione della dimensionalità e nell’individuazione delle variabili più rilevanti.

Sebbene non essenziale in questo caso, il PCA è utile per dataset con molte colonne o più di due cluster, migliorando prestazioni e visualizzazione dei dati.

<br>

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_plot = pd.DataFrame(X_pca, columns=['Componente 1', 'Componente 2'])
df_plot['Cluster'] = df['Cluster']
plt.figure(figsize=(8,6))
for cluster in df_plot['Cluster'].unique():
    plt.scatter(
        df_plot[df_plot['Cluster'] == cluster]['Componente 1'],
        df_plot[df_plot['Cluster'] == cluster]['Componente 2'],
        label=f'Cluster {cluster}'
    )
plt.title('Visualizzazione dei Cluster con PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()
plt.grid(True)
plt.show()
```

**Riduce la dimensionalità per visualizzare i cluster in 2D.**

<br>

## 📊 Interpretazione dei Grafici e dei Profili

- **Distribuzione delle variabili**: Mostra come sono distribuite le attivazioni e i consumi.  
- **Evoluzione temporale**: Permette di identificare le tendenze di consumo nel corso dei giorni.  
- **Attivazioni settimanali**: Aiuta a visualizzare i pattern per ambiente.  
- **Correlazione**: Mostra la forza della relazione tra attivazioni e consumo.  
- **Consumo reale vs previsto**: Valuta la qualità del modello predittivo.  
- **Clusterizzazione**: Identifica gruppi con comportamenti simili per raccomandazioni personalizzate.

<br>

## 💡 Raccomandazioni per Profilo

| Profilo | Principali Raccomandazioni |
| :-- | :-- |
| 🔵 Basso Consumo | Mantenere buone pratiche, incentivare energia solare, premi/gamification |
| 🟡 Consumo Bilanciato | Automatizzare spegnimenti, installare sensori di presenza, report comparativi |
| 🟠 Consumo Elevato | Automatizzare luci/elettronica, sensori di presenza specifici, monitorare picchi |
| 🔴 Alto Consumo (Sala/Cucina) | Pianificare pompa fuori picco, uso consapevole dell’illuminazione, suggerire automazione e tariffa oraria |
| 🔴 Alto Consumo (Cucina) | Verificare apparecchiature, controllare orari di utilizzo, incentivare efficienza dell’illuminazione |

<br>

## 🧭 Conclusione

Il progetto consente di identificare i pattern di consumo, prevedere l’uso futuro e raccomandare azioni per una maggiore efficienza energetica, personalizzando le raccomandazioni in base al profilo di utilizzo di ogni abitazione.


<br>

**Nota:**  
Adattare il percorso del file Excel (`file_path`) in base al proprio ambiente.

<br>

**Questa analisi è stata preparata basandosi sulle pratiche di data science applicate al contesto del consumo energetico residenziale e mira a facilitare il processo decisionale per il cliente finale.**







#

##### <p align="center"> Copyright 2024 Mindful-AI-Assistants. Codice rilasciato sotto [licenza MIT](https://github.com/Mindful-AI-Assistants/planet-smart-city-laguna-iot-pucsp/blob/7ac78ed36a9256cbdc0941dbd44fd13b545bc2dd/LICENSE).






