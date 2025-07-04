<br>

 \[**[🇮🇹 Italiano](README.it_IT.md)**\]  \[[🇧🇷 Português](README.pt_BR.md)\] \[[🇺🇸 English](README.md)\]

<br><br><br><br>

## <p align="center">   CDIA Nexus PUC-SP: Centro di Innovazione per Soluzioni Intelligenti di Acqua ed Energia ⚡️  Smart City Laguna IoT, Fortaleza, Brasile 
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

## 🌐 [Panoramica del Progetto]():

<br>

Sviluppato dal **gruppo CDIA della PUC-SP**, questo progetto di estensione mira a ottimizzare i **sistemi intelligenti di gestione delle risorse** nella Smart City Laguna — combinando **tecnologia**, **sostenibilità** e **innovazione comunitaria** per rafforzare le regioni svantaggiate.

Con una solida base di **collaborazione interdisciplinare** e cooperazione internazionale, questa iniziativa collega la scienza dei dati ad applicazioni reali per promuovere città resilienti, inclusive e intelligenti.

<br>

## [Planet Smart City]():

Fondata nel 2015 da [**Giovanni Savio**](https://github.com/user-attachments/assets/e53824a7-19df-464d-a12f-61024ba18a94) e [**Susanna Marchionni**](), Planet Smart City guida il movimento globale per **abitazioni accessibili, intelligenti e sostenibili**. I loro progetti combinano:

- Progettazione urbana avanzata  
- Tecnologia integrata  
- Iniziative di costruzione comunitaria

<br>

## ⚡️ [CDIA Nexus PUC-SP: Hub di Innovazione per Acqua ed Energia Intelligenti]() 


Il **CDIA Nexus** è un'iniziativa del Gruppo di Scienza dei Dati e Intelligenza Artificiale della **PUC-SP**, dedicata allo sviluppo di soluzioni applicate di **IA e IoT** per la **gestione intelligente di acqua ed energia**.

Questo hub di innovazione integra **ricerca applicata, estensione universitaria e impatto sociale**, con l'obiettivo di trasformare le comunità attraverso una tecnologia con uno scopo.

Le soluzioni sviluppate sono applicate in contesti reali, come la **Smart City Laguna** (Fortaleza, Brasile), attraverso progetti in collaborazione con organizzazioni come **Planet Smart City**, **UN-Habitat**, **Starlink**, tra le altre.

Il progetto mira a combinare **sostenibilità, inclusione digitale e innovazione sociale**, promuovendo città più **resilienti, efficienti e incentrate sulle persone**.

<br>

## 💥 [Dal Codice all’Intuizione](): Analisi dei Dati e Supporto alle Decisioni:

<br>

## [Obiettivo del Progetto]():

Sviluppare una soluzione basata su scienza dei dati e intelligenza artificiale per **monitorare, prevedere e ottimizzare il consumo di energia elettrica in una casa intelligente** (Smart City Laguna). Il progetto simula i dati dei sensori per stanza e utilizza il machine learning per anticipare i modelli di consumo e proporre azioni di risparmio.

<br>

## [Dataset](): 

#### ☞ [Tocca qui]() per accedere al dataset! 

È stato utilizzato un **dataset simulato**, contenente registrazioni giornaliere con le seguenti variabili:

- `Date`: Giorno della misurazione  
- `KW/H`: Consumo totale di energia in kWh  
- `Room1`, `Room2`, `LivingRoom`, `Kitchen`, `Pool`: Numero di attivazioni dei sensori per stanza  
- `SolarGeneration`: Energia generata dai pannelli solari (simulata)

<br>

## [Domanda di Business]():

<br>

> “Come possiamo prevedere il consumo energetico giornaliero in base al comportamento specifico per stanza e, da lì, proporre misure automatizzate per il risparmio e l’efficienza energetica?”

<br>

## [Metodologia e Fasi]():

<br><br>

1. [**Importazione e visualizzazione dei dati**]()  

Lettura del foglio di calcolo utilizzando `pandas` e validazione dei formati.

<br><br>


2. [***Preprocessing***]()   
   - Conversione della colonna `Date` nel formato `datetime`  
   - Creazione della variabile `OrdinalDay` per la modellazione  
   - Calcolo del consumo medio per attivazione per stanza  
   - Simulazione della generazione solare e proiezione del consumo futuro
   - 

<br><br>


3. [***Modellazione Predittiva***]()

È stato addestrato un modello di **Regressione Lineare** per stimare il consumo (`KW/H`) in base al totale delle attivazioni per stanza. Include anche la previsione per il giorno successivo.


<br><br>


4. [***Visualizzazioni***]() 


   - Time series plots with `matplotlib`/`seaborn`  
   - Ranking of rooms with highest consumption  
   - Activation patterns by cluster  
   - Interactive dashboard using Streamlit for real-time monitoring (optional)
     

 <br><br>
 

5. [***Esportazione del Report***]()  

Generazione automatica di report in PDF con dati rilevanti, grafici e previsioni.


<br><br>


## [Risultati]():

- Il modello di regressione ha mostrato una buona capacità di prevedere il consumo in base all’attività nelle stanze  
- **Soggiorno** e **Cucina** sono stati identificati come le aree a maggiore impatto  
- La **Piscina**, sebbene attivata raramente, ha mostrato un consumo medio elevato per attivazione — indicando spreco; è stata rimossa dal modello, poiché il progetto Laguna è rivolto all’edilizia sociale e non prevede piscine  
- La generazione solare può compensare significativamente il consumo durante le ore di punta, se gestita correttamente

<br>

## [Conclusioni e Raccomandazioni]():

- **Automatizzare lo spegnimento dell'energia** nelle aree ad alto utilizzo come soggiorno e cucina per ottenere risparmi immediati  
- **Programmare l'utilizzo della piscina** per mitigare picchi di consumo non necessari  
- **Sfruttare la generazione solare** per bilanciare l'uso degli elettrodomestici durante le ore di massima produzione  
- **Implementare avvisi** quando vengono superati gli obiettivi di consumo giornalieri

<br>

## [Deliverables]():

- Applicazione Streamlit per il monitoraggio in tempo reale dei sensori  
- Report in PDF con metriche di consumo e raccomandazioni  
- Notebook contenente l'intera pipeline dei dati, modello predittivo e analisi visive

<br>

## [Funzionalità]():

- Dashboard in tempo reale che mostra i dati dei sensori per stanza  
- Previsione del consumo energetico giornaliero tramite Regressione Lineare  
- Sensori simulati per stanza (Stanza1, Stanza2, Soggiorno, Cucina)  
- Obiettivo di consumo giornaliero con sistema di avvisi  
- Sistema di aggiornamento automatico usando `streamlit_autorefresh`  
- Clustering dei pattern di utilizzo tramite KMeans + PCA  
- Esportazione del report in PDF  
- Confronto con la generazione solare simulata

## [Tecnologie Utilizzate]():

- Python  
- Pandas e NumPy – elaborazione e analisi dei dati  
- Scikit-learn – regressione lineare e KMeans  
- Matplotlib, Seaborn e Plotly – visualizzazioni  
- Streamlit – dashboard interattiva  
- FPDF – generazione di report PDF  
- Pillow – rendering delle immagini della dashboard

<br>

## [Struttura del Progetto]():

<br>

laguna_city_digital/  
├── app.py                      # Applicazione principale Streamlit  
├── consumo_model.pkl          # Modello predittivo addestrato  
├── cluster_model.pkl          # Modello KMeans addestrato  
├── dados/  
│   └── Consumo_de_Energia_Analise.xlsx  # Dati simulati per stanza  
├── relatorios/  
│   └── relatorio_consumo_YYYY-MM-DD.pdf  
├── imagens/  
│   ├── grafico_pca.png  
│   ├── heatmap_cluster.png  
│   └── grafico_regressao.png  
└── README.md


<br>

## [Esempi di Visualizzazione]():

- [***Cluster PCA***]()  
  Distribuzione dei modelli di utilizzo per profilo energetico

- [***Heatmap delle Attivazioni***]()  
  Percentuale di utilizzo per stanza

- [***Grafico Reale vs Predetto***]()  
  Valutazione dell'accuratezza della previsione

<br>

## [Prestazioni del Modello]():

- `R²`: 0,70  
- `RMSE`: 11.528,06  
- `Stanza più influente`: Soggiorno (28,21%)

<br>

## [Conclusioni Finali]():

#### Obiettivi di consumo personalizzati  
#### Sistema di allerta in tempo reale  
- Supporto alla sostenibilità energetica urbana  
- Base scalabile per l’implementazione completa di una Smart City

<br>

📌 ***Questa analisi è stata sviluppata utilizzando pratiche di data science applicate al consumo energetico residenziale, con l'obiettivo di supportare il processo decisionale dell'utente finale.***

<br>

## [Panoramica della Presentazione]():

**CDIA Nexus** è il progetto finale di estensione accademica e sociale del **Gruppo di Scienza dei Dati e Intelligenza Artificiale della PUC-SP**, incentrato sull'applicazione di **IoT e IA** per sistemi intelligenti di **acqua ed energia** nella **Smart City Laguna**, uno sviluppo urbano pionieristico a Fortaleza, in Brasile.

Questa iniziativa è stata sviluppata in collaborazione con **Planet Smart City**, **UN-Habitat** e **Starlink**, allineata agli **Obiettivi di Sviluppo Sostenibile (SDGs)** delle Nazioni Unite e impegnata in **innovazione sociale, inclusione digitale e intelligenza ambientale**.

<br>

[I punti salienti della presentazione]():

- Una **Dashboard integrata per il monitoraggio di acqua ed energia**  
- Analisi predittiva con modelli di IA  
- Coinvolgimento della comunità attraverso strategie guidate dai dati  
- Approfondimenti sull’implementazione grazie alla **connettività Starlink** e all’infrastruttura di Planet

<br>

#### [“Dati per il Bene. Innovazione con Significato.”]()

<br>

🌟 Collaboratore chiave: [**Stefano Buono**](https://github.com/user-attachments/assets/e53824a7-19df-464d-a12f-61024ba18a94), fisico e imprenditore, ex ricercatore del CERN e fondatore di AAA (venduta a Novartis), attualmente Presidente di LIFTT e CEO di **Newcleo** (innovazione nucleare pulita).

<br>

➢ [Visita Planet Smart City - Ufficiale](https://planetsmartcity.com/) 🇮🇹

➢ [Visita Planet Smart City - Brasile](https://planetsmartcity.com.br) 🇧🇷

➢ [Visita Planet Smart City - India](https://planetsmartcity.in/) 🇮🇳


<br>

## [Il Progetto Laguna: Innovazione Sociale Intelligente]():

Situata a [**São Gonçalo do Amarante**](), Ceará, Fortaleza, Brasile; **Smart City Laguna** è una [città intelligente di punta di Planet]() in Brasile, con oltre [**60 soluzioni intelligenti**](), tra cui:

- Wi-Fi pubblico e infrastruttura IoT  
- Mobilità urbana sostenibile e illuminazione  
- Drenaggio delle acque piovane con pavimentazioni permeabili  
- Programmi culturali, educativi e di governance

<br>

## [Al centro di questo ecosistema]():  

[***Il Community Manager***]() — un professionista formato dedicato a:

- Mobilitare la governance partecipativa  
- Promuovere workshop, educazione e coinvolgimento  
- Coltivare la coesione sociale e la gestione a lungo termine

<br>  

## 🌎 [Partnerships Globali]():

Un ringraziamento speciale a [Pedro Braida Neto](https://www.linkedin.com/in/pedro-braida-neto-95a047174/), CEO di Proptech Brazil, per guidare con empatia, rispetto e integrità. Il modo in cui ti dedichi agli altri fa davvero la differenza.

<br>

#### [Scrivi a Pedro](mailto:pedro@flexautomation.com.br) 📲

<br>

Esprimiamo la nostra più sentita gratitudine alle organizzazioni e alle persone che hanno reso possibile l'implementazione del CDIA PUC-SP. Un ringraziamento speciale a:

<br>

| [**Organizzazione**]()         | [**Contributo**]()                                         |
|---------------------------|--------------------------------------------------------------|
| **Nazioni Unite (ONU)**   | Finanziamento per l'acquisto di pannelli solari             |
| **PUC-SP (CDIA)**         | Progettazione e implementazione di IoT e Intelligenza Artificiale |
| **UN-Habitat**            | Supporto tecnico e quadri etici                              |
| **Starlink**              | Infrastruttura internet satellitare                          |
| **Planet Smart City**     | Sviluppo urbano e supporto in loco                           |
| **Proptech Brazil**       | Implementazione locale e supporto strategico                 |

<br>

### [Esprimiamo inoltre la nostra gratitudine a]():

- I leader locali e i membri della comunità per la fiducia e la continua collaborazione.  
- Il team tecnico multidisciplinare per la dedizione a soluzioni innovative e sostenibili.  
- Tutti coloro che hanno contribuito, direttamente o indirettamente, a dare vita a questa visione.

<br>

##### [Insieme](), questi partner incarnano un approccio integrato per il raggiungimento degli **Obiettivi di Sviluppo Sostenibile**, in particolare nelle regioni emergenti. 💙🌎

<br>

## ⚡ [CDIA PUC-São Paulo Module: Water & Energy Systems]() 💦

The **Water & Energy Module** designed by CDIA focuses on the use of **IoT and AI** for resource optimization. Key features:

- Smart sensors for consumption monitoring  
- AI-driven dashboards with predictive alerts  
- Visualizations for community awareness  
- Scalable resource management models

<br>
  

## 🧑🏼‍🚀 [Team Members]():

| Name                    | Role                                             |
|-------------------------|--------------------------------------------------|
| **Andson Ribeiro**       | [Github](https://github.com/andsonandreribeiro09) - [Contact]() |
| **Fabiana 🧬 Campanari** | [Github](https://github.com/FabianaCampanari) - [Contact Hub](https://linktr.ee/fabianacampanari)   |
| **Leonardo X Fernandes** |   [Github](https://github.com/LeonardoXF)  - [Contact]()  |
|  **Pedro Vyctor Almeida** |  [Github](https://github.com/ppvyctor) - [Contact]()    |

<!--
| **Leonardo X Fernandes** |   [Github](https://github.com/LeonardoXF)  - [Contact]()  |
-->
<br>


💙 Tutti i membri hanno contribuito in modo collaborativo nelle aree tecniche e creative. Fabiana 🧬 Campanari ha anche guidato l'**identità del progetto e il linguaggio visivo**.

<br>

## [Innovazioni e Attività]():

- Installazione di **sensori IoT** per il monitoraggio di acqua ed energia  
- Sviluppo di **cruscotti predittivi** e sistemi di allerta  
- Co-creazione di un'**interfaccia di visualizzazione dati** per i residenti  
- Implementazione di **progetti pilota di energia solare** supportati da UN-Habitat  
- Analisi in tempo reale per la **pianificazione delle risorse e la sostenibilità**

<br>

## [Risultati di Apprendimento]():

***Il team ha acquisito esperienza pratica in***:

- **Design thinking + metodologie partecipative**  
- **Ricerca sul campo in infrastrutture urbane**  
- **Machine learning e modellazione dei dati**  
- **Prototipazione e integrazione di sistemi**  
- Fornitura di soluzioni che riflettono i **bisogni reali della comunità**

<br>

## [Documentazione Visiva]():

<br>

📷 [**Galleria Fotografica**]()
- `drone_view_laguna_2025.jpg` – Vista aerea della città  
- `team_workshop_on_site.jpeg` – Attività sul campo con i residenti  
- `solar_panels_community.jpeg` – Installazione solare supportata dall'ONU  
- `iot_dashboard_mockup.png` – Anteprima del design del cruscotto  

<br>

## [**Presentazioni**]()
- `CDIA_Final_Pitch.pdf` – Principali intuizioni e risultati  
- `UN_SolarInvestment_Laguna.pptx` – Presentazione per gli stakeholder  
- `IoT_Architecture_Prototype.pptx` – Architettura dei sensori e flusso dei dati

<br> 

## [Ringraziamenti]():

Rivolgiamo un sincero ringraziamento a **Pedro di Proptech**, la cui guida e competenza sono state fondamentali durante tutto il progetto. Il suo supporto e la sua visione sono stati pilastri essenziali del nostro percorso di sviluppo.

<br>

## [Dataset Utilizzato]

#### ☞ [clicca qui](https://github.com/Mindful-AI-Assistants/planet-smart-city-laguna-iot-pucsp/blob/fbdedce4202703e7fea05d247d13ec58d94edd80/2-CRISP-DM%20-%20Project%20Smart%20City%20Laguna/%F0%9F%87%BA%F0%9F%87%B8CRISP-DM_Project_Smart_City_Laguna/Consumo_de_Energia_Analise.xlsx) per ottenere il dataset

[È stato utilizzato un dataset **simulato**, contenente registrazioni giornaliere con le seguenti variabili]():

- `Data`: Giorno della misurazione  
- `KW/H`: Consumo totale di energia in kWh  
- `Quarto1`, `Quarto2`, `Sala`, `Cozinha`, `Piscina`: Numero di attivazioni dei sensori in ogni stanza  
- `Geração Solar`: Energia generata dai pannelli solari (simulata)

<br>

## [**Modellazione Predittiva**]():  

È stato addestrato un modello di [**Regressione Lineare**] per stimare il consumo (`KW/H`) in base al numero totale di attivazioni per stanza. È stata inoltre implementata la previsione per il giorno successivo.

<br>

[**Visualizzazioni**]():  
- Grafici a serie temporali con `matplotlib`/`seaborn`.  
- Classifica delle stanze con maggiore consumo.  
- Rappresentazioni delle attivazioni per cluster.  
- Dashboard interattiva Streamlit per la visualizzazione in tempo reale (opzionale).

<br>

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



<br><br>

## 💌 [Let the data flow... Ping Us]()


- 👩🏻‍🚀 **Fabiana Campanari** - [Shoot me an email](mailto:fabicampanari@proton.me)
  
- 🧑🏼‍🚀 **PedroVyctor** - [Hit me up by email](mailto:pedro.vyctor00@gmail.com)

- 👨🏽‍🚀 **Andson Ribeiro** - [Slide into my inbox]()



<br> 


#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />


<br><br>

<p align="center">  ────────────── ⊹🔭๋ ──────────────

<!--
<p align="center">  ────────────── 🛸๋*ੈ✩* 🔭*ੈ₊ ──────────────
-->

<br>

<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>
  

  




#

##### <p align="center"> Copyright 2024 Mindful-AI-Assistants. Codice rilasciato sotto [licenza MIT](https://github.com/Mindful-AI-Assistants/planet-smart-city-laguna-iot-pucsp/blob/7ac78ed36a9256cbdc0941dbd44fd13b545bc2dd/LICENSE).






