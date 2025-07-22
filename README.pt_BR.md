
 \[[🇮🇹 Italiano](README.it_IT.md)\] \[**[🇧🇷Português](README.pt_BR.md)**\] \[[🇺🇸English](README.md)\]

<br><br><br>

## <p align="center">   CDIA Nexus PUC-SP: Centro de Inovação para Soluções Inteligentes de Água e Energia ⚡️  Smart City Laguna IoT, Fortaleza, Brasil 
#### <p align="center"> ***Projeto para monitoramento, previsão e otimização do consumo energético em uma casa inteligente, utilizando [IoT]() e [IA](). Desenvolvido no contexto da [Smart City Laguna]() – CDIA PUC-SP***.

  <br><br>
 


<p align="center">
   <a href="https://github.com/sponsors/Mindful-AI-Assistants">
    <img src="https://img.shields.io/badge/Sponsor-Mindful%20AI%20Assistants-brightgreen?logo=GitHub&style=flat-square">
  </a>
</p>



<br><br>


#### <p align="center"> ***Em colaboração com [Planet]() Smart City, [PUC-SP]() - Ciência de Dados & IA, [ONU](0 Objetivos de Desenvolvimento Sustentável ODS), [Starlink]() e [Proptech]() Brasil***


 <br><br>

 
<!-- VIDEO -->

 
https://github.com/user-attachments/assets/24546329-4480-4f48-9948-b53ba2ec17fb


#### 📺 [Assista em Full HD su YouTube](https://youtu.be/WmtFxV5G8Fg)


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


## 🌐 [Visão Geral do Projeto]():

<br>

Desenvolvido pelo **grupo CDIA da PUC-SP**, este projeto de extensão tem como objetivo otimizar **sistemas inteligentes de gestão de recursos** na Smart City Laguna — combinando **tecnologia**, **sustentabilidade** e **inovação comunitária** para empoderar regiões subatendidas.

Com uma base sólida em **colaboração interdisciplinar** e cooperação internacional, esta iniciativa conecta ciência de dados com aplicações do mundo real para fomentar cidades resilientes, inclusivas e inteligentes.

<br>

## [Planet Smart City]():

Fundada em 2015 por [**Giovanni Savio**](https://github.com/user-attachments/assets/e53824a7-19df-464d-a12f-61024ba18a94) e [**Susanna Marchionni**](), a Planet Smart City lidera o movimento global por **habitação acessível, inteligente e sustentável**. Seus projetos combinam:

- Design urbano avançado  
- Tecnologia integrada  
- Iniciativas de construção comunitária

<br>

 ## ⚡️ [CDIA Nexus PUC-SP: Polo de Inovação em Água e Energia Inteligente]() 💦

O **CDIA Nexus** é uma iniciativa do Grupo de Ciência de Dados e Inteligência Artificial da **PUC-SP**, dedicada ao desenvolvimento de soluções aplicadas de **IA e IoT** para a **gestão inteligente de água e energia**.

Esse polo de inovação integra **pesquisa aplicada, extensão universitária e impacto social**, com foco na transformação de comunidades por meio de tecnologia com propósito.

As soluções desenvolvidas são aplicadas em contextos reais, como a **Smart City Laguna** (Fortaleza, Brasil), por meio de projetos em parceria com organizações como **Planet Smart City**, **ONU-Habitat**, **Starlink**, entre outras.

O projeto busca combinar **sustentabilidade, inclusão digital e inovação social**, promovendo cidades mais **resilientes, eficientes e centradas nas pessoas**.

<br>

## 💥 [Do Código ao Insight](): Análise de Dados e Apoio à Decisão:

<br>

## [Objetivo do Projeto]():

Desenvolver uma solução baseada em ciência de dados e IA para **monitorar, prever e otimizar o consumo de energia elétrica em uma casa inteligente** (Smart City Laguna). O projeto simula dados de sensores por cômodo e utiliza aprendizado de máquina para antecipar padrões de consumo e propor ações de economia.

<br>

## [Conjunto de Dados Utilizado]():

Foi utilizado um **conjunto de dados simulado**, contendo registros diários com as seguintes variáveis:

- `Data`: Dia da medição  
- `KW/H`: Consumo total de energia em kWh  
- `Quarto1`, `Quarto2`, `Sala`, `Cozinha`, `Piscina`: Número de ativações de sensores por cômodo  
- `GeracaoSolar`: Energia gerada por painéis solares (simulado)

<br>

## [Pergunta de Negócio]():

<br>

> “Como podemos prever o consumo diário de energia com base no comportamento por cômodo e, a partir disso, propor medidas automáticas para economia e eficiência energética?”

<br>

## [Metodologia e Etapas]():

<br>

[1](). **Importação e visualização dos dados**  

Leitura da planilha com `pandas` e validação dos formatos.

<br>

[2](). ***Pré-processamento***

- Conversão da coluna `Date` para o formato `datetime`  
- Criação da variável `OrdinalDay` para modelagem  
- Cálculo do consumo médio por ativação por cômodo  
- Simulação da geração solar e projeção de consumo futuro

<br>

[3](). ***Modelagem Preditiva***

Um modelo de **Regressão Linear** foi treinado para estimar o consumo (`KW/H`) com base no total de ativações por cômodo. Também inclui previsão para o dia seguinte.

<br>

[4](). ***Visualizações*** 
   
- Gráficos de séries temporais com `matplotlib`/`seaborn`  
- Ranking dos cômodos com maior consumo  
- Padrões de ativação por cluster  
- Dashboard interativo com Streamlit para monitoramento em tempo real (opcional)

<br>

[5](). ***Exportação de Relatórios***  

Geração automática de relatórios em PDF com dados relevantes, gráficos e previsões.

<br>

## [Resultados]():

- O modelo de regressão apresentou boa capacidade de prever o consumo com base na atividade por cômodo  
- **Sala** e **Cozinha** foram identificadas como as áreas de maior impacto  
- A **Piscina**, embora raramente ativada, apresentou alto consumo médio por ativação — indicando desperdício; foi removida do modelo, já que o projeto Laguna é voltado à habitação social e não inclui piscinas  
- A geração solar pode compensar significativamente o consumo nos horários de pico, se for bem gerenciada

<br>

## [Conclusões e Recomendações]():

- **Automatizar desligamentos** em áreas de alto uso como sala e cozinha, para gerar economia imediata  
- **Agendar o uso da piscina** para mitigar picos de consumo desnecessários  
- **Aproveitar a geração solar** para equilibrar o uso de aparelhos nos horários de maior produção  
- **Implementar alertas** quando as metas diárias de consumo forem ultrapassadas

<br>

## [Entregáveis]():

- Aplicativo Streamlit para monitoramento em tempo real dos sensores  
- Relatório em PDF com métricas de consumo e recomendações  
- Notebook contendo todo o pipeline de dados, modelo preditivo e análises visuais

<br>

## [Funcionalidades]():

- Dashboard em tempo real exibindo dados dos sensores por cômodo  
- Previsão diária de consumo energético via Regressão Linear  
- Sensores simulados por cômodo (Quarto1, Quarto2, Sala, Cozinha)  
- Meta diária de consumo com sistema de alertas  
- Sistema de atualização automática usando `streamlit_autorefresh`  
- Clusterização de padrões de uso com KMeans + PCA  
- Exportação de relatórios em PDF  
- Comparação com geração solar simulada

<br>

## [Tecnologias Utilizadas]():

- Python  
- Pandas e NumPy – processamento e análise de dados  
- Scikit-learn – regressão linear e KMeans  
- Matplotlib, Seaborn e Plotly – visualizações  
- Streamlit – dashboard interativo  
- FPDF – geração de relatórios em PDF  
- Pillow – renderização de imagens no dashboard

<br>

## [Estrutura do Projeto]():

<br>

laguna_city_digital/  
├── app.py                      # Aplicativo principal em Streamlit  
├── consumo_model.pkl          # Modelo de previsão treinado  
├── cluster_model.pkl          # Modelo KMeans treinado  
├── dados/  
│   └── Consumo_de_Energia_Analise.xlsx  # Dados simulados por cômodo  
├── relatorios/  
│   └── relatorio_consumo_YYYY-MM-DD.pdf  
├── imagens/  
│   ├── grafico_pca.png  
│   ├── heatmap_cluster.png  
│   └── grafico_regressao.png  
└── README.md

<br>

## [Exemplos de Visualizações]():

- [***Clusterização com PCA***]()  
  Distribuição de padrões de uso por perfil energético

- [***Mapa de Calor de Ativações***]()  
  Percentual de uso por cômodo

- [***Gráfico Real vs Previsto***]()  
  Avaliação da precisão da previsão

<br>

## [Desempenho do Modelo]():

- `R²`: 0,70  
- `RMSE`: 11.528,06  
- `Cômodo com maior influência`: Sala (28,21%)

<br>

## [Conclusões Finais]():

#### Metas personalizadas de consumo  
#### Sistema de alertas em tempo real  
- Apoio à sustentabilidade energética urbana  
- Base escalável para implantação completa de uma Smart City

<br>

📌 ***Esta análise foi desenvolvida com base em práticas de ciência de dados aplicadas ao consumo de energia residencial, com o objetivo de apoiar a tomada de decisões pelo usuário final.***

<br>

## [Apresentação Geral]():

**CDIA Nexus** é o projeto final de extensão acadêmica e social do **Grupo de Ciência de Dados e Inteligência Artificial da PUC-SP**, com foco na aplicação de **IoT e IA** para sistemas inteligentes de **água e energia** na **Smart City Laguna**, um desenvolvimento urbano pioneiro em Fortaleza, Brasil.


Essa iniciativa foi desenvolvida em parceria com a **Planet Smart City**, **ONU-Habitat** e **Starlink**, alinhada aos **Objetivos de Desenvolvimento Sustentável (ODS) da ONU** e comprometida com a **inovação social, inclusão digital e inteligência ambiental**.

<br>

[Principais destaques da apresentação]():

- Um **Dashboard Integrado de Monitoramento de Água e Energia**  
- Análises preditivas com modelos de IA  
- Engajamento comunitário com base em estratégias orientadas por dados  
- Insights de implantação utilizando **conectividade via Starlink** e infraestrutura Planet

<br>

#### [“Data for Good. Inovação com Significado.”]()

<br>

🌟 Contribuição-chave: [**Stefano Buono**](https://github.com/user-attachments/assets/e53824a7-19df-464d-a12f-61024ba18a94), físico e empreendedor, ex-pesquisador do CERN e fundador da AAA (vendida para a Novartis), atualmente Presidente da LIFTT e CEO da **Newcleo** (inovação nuclear limpa).

<br>

➢ [Visite Planet Smart City - Oficial](https://planetsmartcity.com/) 🇮🇹

➢ [Visite Planet Smart City - Brasil](https://planetsmartcity.com.br) 🇧🇷

➢ [Visite Planet Smart City - Índia](https://planetsmartcity.in/) 🇮🇳


<br>

## [O Projeto Laguna: Inovação Social Inteligente]():

Localizada em [**São Gonçalo do Amarante**](), Ceará, Fortaleza, Brasil; a **Smart City Laguna** é a [cidade inteligente modelo da Planet]() no Brasil, contando com mais de [**60 soluções inteligentes**](), incluindo:

- Wi-Fi público e infraestrutura IoT  
- Mobilidade urbana e iluminação sustentáveis  
- Drenagem de águas pluviais com pavimentos permeáveis  
- Programas culturais, educacionais e de governança

<br>

## [No centro desse ecossistema]():

[***O Gestor Comunitário*** ]() — um profissional capacitado, dedicado a:

- Mobilizar a governança participativa  
- Promover oficinas, educação e engajamento  
- Fortalecer a coesão social e o cuidado de longo prazo

<br>  

## 🌎 [Parcerias Globais]():

Agradecimento especial a [Pedro Braida Neto](https://www.linkedin.com/in/pedro-braida-neto-95a047174/), CEO da Proptech Brasil, por liderar com empatia, respeito e integridade. A forma como você apoia os outros realmente faz toda a diferença.

<br>

#### [Fale com Pedro](mailto:pedro@flexautomation.com.br) 📲

<br>

Expressamos nossa sincera gratidão às organizações e indivíduos que tornaram possível a implementação do CDIA PUC-SP. Agradecimento especial para:

<br>

| [**Organização**]()           | [**Contribuição**]()                                      |
|------------------------------|------------------------------------------------------------|
| **Nações Unidas (ONU)**      | Financiamento para aquisição de painéis solares            |
| **PUC-SP (CDIA)**            | Design e implementação de IoT e IA                         |
| **ONU-Habitat**              | Apoio técnico e estruturação ética                         |
| **Starlink**                 | Infraestrutura de internet via satélite                   |
| **Planet Smart City**        | Desenvolvimento urbano e suporte local                    |
| **Proptech Brasil**          | Implementação local e suporte estratégico                 |

<br>

### [Também agradecemos a]():

- Líderes locais e membros da comunidade pela confiança e colaboração contínua.  
- A equipe técnica multidisciplinar pela dedicação a soluções inovadoras e sustentáveis.  
- Todos que contribuíram, direta ou indiretamente, para tornar essa visão realidade.


<br>

##### [Juntos](), esses parceiros representam uma abordagem integrada para alcançar os **Objetivos de Desenvolvimento Sustentável**, especialmente em regiões emergentes. 💙🌎

<br>

## ⚡ [Módulo CDIA PUC-São Paulo: Sistemas de Água e Energia]() 💦

O **Módulo de Água e Energia** desenvolvido pelo CDIA tem foco no uso de **IoT e IA** para otimização de recursos. Principais funcionalidades:

- Sensores inteligentes para monitoramento de consumo  
- Dashboards com alertas preditivos baseados em IA  
- Visualizações para conscientização comunitária  
- Modelos de gestão de recursos escaláveis

<br>

## 🧑🏼‍🚀 [Membros da Equipe]():

| Nome                    | Função                                             |
|-------------------------|----------------------------------------------------|
| **Andson Ribeiro**       | [Github](https://github.com/andsonandreribeiro09) - [Contato]() |
| **Fabiana 🧬 Campanari** | [Github](https://github.com/FabianaCampanari) - [Hub de Contato](https://linktr.ee/fabianacampanari) |
| **Leonardo X Fernandes** | [Github](https://github.com/LeonardoXF) - [Contato]()  |
| **Pedro Vyctor Almeida** | [Github](https://github.com/ppvyctor) - [Contato]()    |

<!--
| **Leonardo X Fernandes** | [Github](https://github.com/LeonardoXF) - [Contato]()  |
-->

<br>

💙 Todos os membros contribuíram colaborativamente nas áreas técnicas e criativas. Fabiana 🧬 Campanari também liderou a **identidade e linguagem visual do projeto**.

<br>

## [Inovações & Atividades]():

- Instalação de **sensores IoT** para monitoramento de água e energia  
- Desenvolvimento de **dashboards preditivos** e sistemas de alerta  
- Cocriação de uma **interface de visualização de dados** para os moradores  
- Implementação de **pilotos de energia solar** com apoio da ONU-Habitat  
- Análises em tempo real para **planejamento de recursos e sustentabilidade**

<br>

## [Resultados de Aprendizagem]():

***A equipe adquiriu experiência prática em***:

- **Design thinking + metodologias participativas**  
- **Pesquisa de campo em infraestrutura urbana**  
- **Aprendizado de máquina e modelagem de dados**  
- **Prototipagem e integração de sistemas**  
- Entrega de soluções que refletem as **necessidades reais da comunidade**

<br>


## [Documentação Visual]():

<br>

📷 [**Galeria de Fotos**]()
- `drone_view_laguna_2025.jpg` – Vista aérea da cidade  
- `team_workshop_on_site.jpeg` – Atividades de campo com moradores  
- `solar_panels_community.jpeg` – Instalação solar com apoio da ONU  
- `iot_dashboard_mockup.png` – Prévia do design do dashboard  

<br>

## [**Apresentações**]()
- `CDIA_Final_Pitch.pdf` – Principais insights e resultados  
- `UN_SolarInvestment_Laguna.pptx` – Apresentação para stakeholders  
- `IoT_Architecture_Prototype.pptx` – Arquitetura de sensores e fluxo de dados

<br> 

## [Agradecimento]():

Agradecemos sinceramente ao **Pedro da Proptech**, cuja orientação e expertise foram fundamentais ao longo deste projeto. Seu apoio e visão foram pilares essenciais em nossa jornada de desenvolvimento.

<br>

<br>

## Conjunto de Dados Utilizado - [clique aqui para acessar o dataset]():

Foi utilizado um [**conjunto de dados simulado**], contendo registros diários com as seguintes variáveis:

- `Data`: Dia da medição  
- `KW/H`: Consumo total de energia em kWh  
- `Quarto1`, `Quarto2`, `Sala`, `Cozinha`, `Piscina`: Número de ativações de sensores em cada cômodo  
- `Geração Solar`: Energia gerada por painéis solares (simulado)

<br>

## [**Modelagem Preditiva**]():

Um modelo de [**Regressão Linear**] foi treinado para estimar o consumo (`KW/H`) com base nas ativações totais por cômodo. Também foi implementada a previsão para o dia seguinte.

<br>

[**Visualizações**]():  
   - Gráficos de séries temporais com `matplotlib`/`seaborn`.  
   - Ranking dos cômodos com maior consumo.  
   - Representações de ativações por cluster.  
   - Dashboard interativo com Streamlit para visualização em tempo real (opcional).

<br>

## 📓 [Pipeline de Código]():


<br>

### **[Célula 1]() — Importação das bibliotecas**

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

### **[Célula 2]() — Leitura dos dados**

```python
# Altere o caminho conforme seu ambiente
file_path = "/Users/fabicampanari/Desktop/Project Planet Smart City Laguna/2-CRISP-DM - Project Smart City Laguna/🇧🇷 CRISP-DM_Projeto_Smart_City_Laguna/Consumo_de_Energia_Analise.xlsx"
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
print(sheet_names)
df = xls.parse('Sheet1')
print(df.head())
df.info()
```


<br>

### **[Célula 3]() — Pré-processamento de datas**

```python
meses_pt = {
    'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04',
    'mai': '05', 'jun': '06', 'jul': '07',
}
df['Data'] = df['Data'].astype(str)
df['Data'] = df['Data'].str.lower().replace(meses_pt, regex=True)
df['Data'] = pd.to_datetime(df['Data'] + '/2025', format='%d/%m/%Y')
```


<br>

### **[Célula 4]() — Estatísticas descritivas e correlação**

```python
summary = df.describe()
correlation = df.corr(numeric_only=True)
print(summary)
print(correlation)
```


<br>

### [**Célula 5:  PLOT 1]() - Distribuição das variáveis**

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
cols = df.columns[1:7]
for i, col in enumerate(cols):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=10)
    axes[i].set_title(f'Distribuição - {col}')
    axes[i].set_xlabel(col)
plt.tight_layout()
plt.suptitle("Distribuição das Variáveis", fontsize=16, y=1.02)
plt.show()
```

<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/d4bb66f3-c8aa-42f1-a36e-6ef32a55fbdb"/>


<br><br>

### **[Célula 6: PLOT 2]() - Evolução do consumo total ao longo do tempo**

```python
plt.figure(figsize=(14, 6))
plt.plot(df['Data'], df['KW/H'], label='Consumo Total (KW/H)', color='blue', linewidth=2)
plt.title('Evolução do Consumo Total de Energia')
plt.xlabel('Data')
plt.ylabel('KW/H')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/af0b1499-d4ea-44d6-9e13-30e2d72d643b"/>

<br><br>

### **[Célula 7: Plot 3]() — Agrupamento semanal e Acionamentos semanais por cômodo**

```python
df['Semana'] = df['Data'].dt.to_period('W').apply(lambda r: r.start_time)
df_semana = df.groupby('Semana')[['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'Piscina']].sum()
df_semana.plot(figsize=(12, 6), marker='o')
plt.title('Acionamentos Semanais por Cômodo')
plt.ylabel('Número de Acionamentos')
plt.xlabel('Semana')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/d31b2df3-8a5c-443a-a16d-60b8f3c53792" />

<br><br>

### **[Célula 8: PLOT 4]() - Correlação entre acionamentos e consumo**

```python
correlacoes = df[['KW/H', 'Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'Piscina']].corr()['KW/H'][1:]
plt.figure(figsize=(10, 5))
sns.barplot(x=correlacoes.index, y=correlacoes.values, palette='Oranges_r')
plt.title('Correlação entre Acionamentos e Consumo de Energia (kWh)')
plt.ylabel('Correlação')
plt.xlabel('Cômodo')
plt.tight_layout()
plt.show()
```

<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/d299ee06-4acc-4f4c-be50-bd82858b95c4" />

<br><br>

### **[Célula 9]() — Modelagem preditiva (Regressão Linear) e avaliação**

```python
X = df[['Quarto1', 'Quarto2', 'Sala', 'Cozinha']]
y = df['KW/H']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Erro quadrático médio (MSE):", round(mse, 2))
print("Coeficiente de determinação (R²):", round(r2, 2))
```


<br>

### **[Célula 10: PLOT 5]() - Consumo real vs previsto**

```python
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Consumo real (kWh)")
plt.ylabel("Consumo previsto (kWh)")
plt.title("Consumo Real vs Previsto")
plt.grid(True)
plt.tight_layout()
plt.show()
```

<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/b2c22d6f-5d6c-47ae-8686-ac9e9d13cf70" />

<br><br>

### **[Célula 11]() — Coeficientes do modelo**

```python
coeficientes = pd.Series(modelo.coef_, index=X.columns)
print("\nContribuição de cada cômodo na previsão (coeficientes):")
print(coeficientes.sort_values(ascending=False))
```

**Mostra o peso de cada cômodo na previsão do consumo.**

<br>

### **[Célula 12]() — Cálculo de percentuais de acionamento por cômodo**

```python
df['Total_acionamentos'] = df[['Quarto1', 'Quarto2', 'Sala', 'Cozinha']].sum(axis=1)
for comodo in ['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'Piscina']:
    df[f'{comodo}_pct'] = df[comodo] / df['Total_acionamentos']
```


<br>

### **[Célula 13: PLOT 6]() - Método do Cotovelo para KMeans**

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
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')
plt.grid(True)
plt.show()
```

<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/777530cb-66e4-4bb1-a13c-94e6f0bcd746" />

<br><br>

### **[Célula 14: PLOT 7]() — KMeans e  Pairplot dos clusters**

```python
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
sns.pairplot(df, hue='Cluster', vars=['Quarto1', 'Quarto2', 'Sala', 'Cozinha'], palette='tab10')
plt.suptitle("Padrões de Uso Agrupados por Cluster", y=1.02)
plt.show()
```

<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/8b833b83-446f-4d35-a070-df58ae5d78b6" />

<br><br>

### **[Célula 15]() — Perfil médio por cluster e nomeação dos perfis**

```python
col_pcts = [f'{c}_pct' for c in ['Quarto1', 'Quarto2', 'Sala', 'Cozinha']]
perfil_clusters = df.groupby('Cluster')[['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'KW/H'] + col_pcts].mean()

def nomear_cluster(row):
    media_kw = df['KW/H'].mean()
    if row['KW/H'] < media_kw * 0.75:
        consumo_total = '🔵 Baixo Consumo'
    elif row['KW/H'] > media_kw * 1.25:
        consumo_total = '🔴 Alto Consumo'
    else:
        consumo_total = '🟡 Consumo Equilibrado'
    altos = []
    for comodo in ['Quarto1', 'Quarto2', 'Sala', 'Cozinha']:
        media_pct = df[f'{comodo}_pct'].mean()
        if row[f'{comodo}_pct'] > media_pct * 1.2:
            altos.append(comodo)
    if consumo_total == '🔵 Baixo Consumo':
        return consumo_total
    if consumo_total == '🟡 Consumo Equilibrado':
        if len(altos) == 0:
            return consumo_total
        else:
            return f"🟠 Consumo Elevado em {', '.join(altos)}"
    if consumo_total == '🔴 Alto Consumo':
        if len(altos) == 0:
            return consumo_total
        else:
            return f"🔴 Alto Consumo (Em {', '.join(altos)})"

perfil_clusters['Perfil'] = perfil_clusters.apply(nomear_cluster, axis=1)
```


<br>

### **[Célula 16]() — Dicionário de recomendações e exibição por cluster**

```python
def mapear_perfil_para_chave(perfil):
    if perfil == '🔵 Baixo Consumo':
        return perfil
    if perfil == '🟡 Consumo Equilibrado':
        return perfil
    if perfil.startswith('🟠 Consumo Elevado'):
        return '🟠 Consumo Elevado'
    if perfil.startswith('🔴 Alto Consumo'):
        if 'Em' in perfil:
            idx = perfil.index('Em') + 3
            texto = perfil[idx:]
            principal = texto.split(',')[^0].strip()
            if principal in ['Sala']:
                return '🔴 Alto Consumo (Sala/Cozinha)'
            elif principal == 'Cozinha':
                return '🔴 Alto Consumo (Cozinha)'
            else:
                return '🔴 Alto Consumo'
        else:
            return '🔴 Alto Consumo'
    return perfil

recomendacoes = {
    '🔵 Baixo Consumo': [
        "✅ Manter boas práticas já adotadas.",
        "🎁 Oferecer recompensas ou descontos (gamificação).",
        "🔋 Incentivar uso de energia solar / microgeração."
    ],
    '🟡 Consumo Equilibrado': [
        "🔌 Automatizar desligamento de equipamentos em horários fixos.",
        "🕵️ Instalar sensores de presença em quartos e sala.",
        "📊 Enviar relatórios semanais de uso comparativo."
    ],
    '🟠 Consumo Elevado': [
        "🛏️ Automatizar luzes e eletrônicos nos cômodos com consumo elevado.",
        "🕵️ Instalar sensores de presença específicos para os cômodos.",
        "📊 Acompanhar o uso para identificar picos desnecessários."
    ],
    '🔴 Alto Consumo (Sala/Cozinha)': [
        "💧 Agendar funcionamento da bomba da Cozinha fora do pico.",
        "💡 Incentivar uso consciente da iluminação e eletrônicos.",
        "🧠 Sugerir automação e adesão à tarifa branca."
    ],
    '🔴 Alto Consumo (Cozinha)': [
        "🍳 Verificar equipamentos de cozinha para consumo excessivo.",
        "⏰ Controlar horários de uso de forno e geladeira.",
        "💡 Incentivar uso eficiente da iluminação."
    ]
}

for cluster_id, row in perfil_clusters.iterrows():
    print(f"\n=== Cluster {cluster_id} - {row['Perfil']} ===")
    print("📊 Perfil médio de consumo (acionamentos e kWh):")
    print(row[['Quarto1', 'Quarto2', 'Sala', 'Cozinha', 'KW/H']])
    print("\n📈 Percentual médio de acionamentos por cômodo (%):")
    print((row[col_pcts] * 100).round(2))
    print("\n💡 Recomendações:")
    chave = mapear_perfil_para_chave(row['Perfil'])
    if chave in recomendacoes:
        for rec in recomendacoes[chave]:
            print("-", rec)
    else:
        print("- Sem recomendações específicas para este perfil.")
```


<br>

### **[Célula 17: PLOT 8]() - Boxplot consumo por cluster**

```python
plt.figure(figsize=(7,5))
sns.boxplot(x='Cluster', y='KW/H', data=df)
plt.title('Distribuição de Consumo (KW/H) por Cluster')
plt.show()
```


<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/d0f56c43-9869-4401-adb4-ce1e0fbcde40" />

<br><br>


### **[Célula 18: PLOT 9]() - Heatmap de percentuais por cluster**

```python
heatmap_data = perfil_clusters[col_pcts] * 100
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Percentual de Acionamentos por Cômodo (%)')
plt.xlabel('Cômodos')
plt.ylabel('Cluster')
plt.show()
```


<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/5087199f-09c1-43d0-95b6-2e2160cbfe5e" />

<br><br>


### **[Célula 19: PLOT 10]() - Radar dos cômodos por cluster**

```python
categorias = ['Quarto1', 'Quarto2', 'Sala', 'Cozinha']
angles = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
angles += angles[:1]
plt.figure(figsize=(10, 8))
for i, row in perfil_clusters.iterrows():
    valores = [row[cat] for cat in categorias]
    valores += valores[:1]
    plt.polar(angles, valores, label=f'Cluster {i}')
plt.xticks(angles[:-1], categorias)
plt.title('Radar dos Cômodos por Cluster')
plt.legend()
plt.show()
```

<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/2f5b77f3-ba95-440f-9bc4-d3076961ab71"/>

<br><br>

### **[Célula 20: PLOT 11]() - Visualização dos clusters com PCA**

<br>

Aplicamos o PCA de forma demonstrativa, mesmo com apenas dois clusters, para mostrar como ele atua na redução de dimensionalidade e na identificação das variáveis mais relevantes.

Embora não seja essencial neste caso, o PCA é útil em cenários com muitas colunas ou mais de dois clusters, ajudando na performance e na visualização dos dados.

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
plt.title('Visualização dos Clusters com PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()
plt.grid(True)
plt.show()
```


<br><br>

 <p align="center">
<img src="https://github.com/user-attachments/assets/82d5e3a8-7c7f-4100-ac35-c8267effc9f0" />

<br><br>


## 📊 Interpretação dos Gráficos e Perfis

- **Distribuição das variáveis**: Mostra como os acionamentos e o consumo se distribuem.
- **Evolução temporal**: Permite identificar tendências de consumo ao longo dos dias.
- **Acionamentos semanais**: Ajuda a visualizar padrões por cômodo.
- **Correlação**: Mostra a força da relação entre acionamentos e consumo.
- **Consumo real vs previsto**: Avalia a qualidade do modelo preditivo.
- **Clusterização**: Identifica grupos de comportamento similares para recomendações personalizadas.

 <br>

## 💡 Recomendações por Perfil

| Perfil | Recomendações Principais |
| :-- | :-- |
| 🔵 Baixo Consumo | Manter boas práticas, incentivar energia solar, recompensas/gamificação |
| 🟡 Consumo Equilibrado | Automatizar desligamentos, instalar sensores de presença, relatórios comparativos |
| 🟠 Consumo Elevado | Automatizar luzes/eletrônicos, sensores de presença específicos, monitorar picos |
| 🔴 Alto Consumo (Sala/Cozinha) | Agendar bomba fora do pico, uso consciente de iluminação, sugerir automação e tarifa branca |
| 🔴 Alto Consumo (Cozinha) | Verificar equipamentos, controlar horários de uso, incentivar eficiência da iluminação |


<br>

## 🧭 Conclusão

O projeto permite identificar padrões de consumo, prever o uso futuro e recomendar ações para maior eficiência energética, personalizando as recomendações conforme o perfil de uso de cada residência.

<br>

**Observação:**
Altere o caminho do arquivo Excel (`file_path`) conforme seu ambiente.

<br>

**EstA Aanalise foi elaborado com base nas práticas de ciência de dados aplicadas ao contexto de consumo energético residencial e tem como objetivo facilitar a tomada de decisão por parte do cliente final.**




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

##### <p align="center"> Copyright 2024 Mindful-AI-Assistants. Código disponibilizado sob a [licença MIT](https://github.com/Mindful-AI-Assistants/planet-smart-city-laguna-iot-pucsp/blob/7ac78ed36a9256cbdc0941dbd44fd13b545bc2dd/LICENSE).
