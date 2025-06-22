import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import random
from streamlit_autorefresh import st_autorefresh
import pydeck as pdk
from fpdf import FPDF
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# CONFIGURAÇÃO STREAMLIT
st.set_page_config(page_title="🌇 Smart City - Simulador Solar", layout="wide")
st_autorefresh(interval=50000, key="refresh")

# BANCO DE DADOS
conn = sqlite3.connect("smartcity_laguna.db", check_same_thread=False)
c = conn.cursor()

# TABELAS
c.execute('''CREATE TABLE IF NOT EXISTS simulacoes (
    timestamp TEXT,
    id_casa INTEGER,
    consumo_kwh REAL,
    geracao_kwh REAL,
    excedente REAL
)''')
c.execute('''CREATE TABLE IF NOT EXISTS consumo_comodos (
    timestamp TEXT,
    id_casa INTEGER,
    Quarto1 INTEGER,
    Quarto2 INTEGER,
    Sala INTEGER,
    Cozinha INTEGER,
    Piscina INTEGER
)''')
conn.commit()

# FUNÇÕES AUXILIARES
def get_estacao(mes):
    return ["verao", "outono", "inverno", "primavera"][(mes % 12) // 3]

def simular_clima():
    return random.choices(
        ["ensolarado", "parcialmente nublado", "nublado", "chuvoso"],
        weights=[0.4, 0.3, 0.2, 0.1]
    )[0]

def fator_climatico(clima):
    return {"ensolarado": 1.0, "parcialmente nublado": 0.7, "nublado": 0.4, "chuvoso": 0.2}[clima]

horas_sol_estacao = {
    "verao": random.uniform(5.5, 6.5),
    "outono": random.uniform(4.0, 5.5),
    "inverno": random.uniform(3.0, 4.5),
    "primavera": random.uniform(4.5, 6.0)
}

casas_coords = [
    {"id": i+1, "lat": -28.482 + random.uniform(-0.005, 0.005), "lon": -48.781 + random.uniform(-0.005, 0.005)}
    for i in range(10)
]

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Relatório de Geração Solar - Smart City Laguna", 0, 1, "C")
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", 0, 0, "C")

def exportar_pdf(df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    for _, row in df.iterrows():
        linha = f"Casa {int(row['id_casa'])} | Consumo: {row['consumo_kwh']:.2f} kWh | Geração: {row['geracao_kwh']:.2f} kWh | Excedente: {row['excedente']:.2f} kWh"
        pdf.cell(0, 10, linha, ln=1)
    pdf.output("relatorio_solar.pdf")

def executar_simulador_solar():
    st.title("🌆 Simulador Solar Smart City Laguna")

    potencia_kwp = st.slider("Potência do sistema (kWp)", 1.0, 10.0, 3.5, 0.5)
    eficiencia = 0.8
    agora = datetime.now()
    hora_atual = agora.hour
    mes = agora.month
    estacao = get_estacao(mes)
    clima = simular_clima()
    fator = fator_climatico(clima)
    horas_sol = horas_sol_estacao[estacao]

    st.markdown(f"""
    🌤️ **Estação**: `{estacao}`  
    🕒 **Hora atual**: `{hora_atual}h`  
    🌦️ **Clima**: `{clima}`  
    ☀️ **Horas de Sol**: `{horas_sol:.2f}h`  
    ⚙️ **Fator climático aplicado**: `{fator * 100:.0f}%`
    """)

    if 6 <= hora_atual <= 18:
        for id_casa in range(1, 11):
            consumo = random.uniform(8, 20)
            geracao = potencia_kwp * horas_sol * eficiencia * fator
            excedente = geracao - consumo
            timestamp = agora.strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO simulacoes VALUES (?, ?, ?, ?, ?)",
                      (timestamp, id_casa, consumo, geracao, excedente))

            dados_comodos = [random.randint(0, 4) for _ in range(5)]  # Quarto1, Quarto2, Sala, Cozinha, Piscina
            c.execute("INSERT INTO consumo_comodos VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (timestamp, id_casa, *dados_comodos))
        conn.commit()
        st.success("✅ Geração registrada com sucesso!")
    else:
        st.warning("⚠️ Fora do horário de captação solar (06h às 18h). Geração pausada.")

    # Dados para análise
    df = pd.read_sql_query("SELECT * FROM simulacoes", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_latest = df.sort_values("timestamp").groupby("id_casa").tail(1)
    df_comodos = pd.read_sql_query("SELECT * FROM consumo_comodos", conn)

    # Criar coluna de timestamp numérico para modelo
    df['hora_timestamp'] = df['timestamp'].map(datetime.timestamp)

    # Função para prever consumo
    def prever_consumo_por_casa(df, horas_a_frente=3):
        previsoes = []
        for casa in df['id_casa'].unique():
            df_casa = df[df['id_casa'] == casa].sort_values('timestamp')
            if len(df_casa) < 5:
                previsoes.append({"id_casa": casa, "consumo_previsto": None})
                continue
            X = df_casa['hora_timestamp'].values.reshape(-1, 1)
            y = df_casa['consumo_kwh'].values
            model = LinearRegression().fit(X, y)
            ultima_hora = df_casa['hora_timestamp'].max()
            horas_futuras = np.array([ultima_hora + 3600 * i for i in range(1, horas_a_frente + 1)]).reshape(-1, 1)
            predicoes = model.predict(horas_futuras)
            consumo_previsto = predicoes.mean()
            previsoes.append({
                "id_casa": casa,
                "consumo_previsto": consumo_previsto,
                "predicoes_horas": predicoes,
                "horas_futuras": horas_futuras.flatten()
            })
        return previsoes

    # Função para prever geração
    def prever_geracao_por_casa(df, horas_a_frente=3):
        previsoes = []
        for casa in df['id_casa'].unique():
            df_casa = df[df['id_casa'] == casa].sort_values('timestamp')
            if len(df_casa) < 5:
                previsoes.append({"id_casa": casa, "geracao_prevista": None})
                continue
            X = df_casa['hora_timestamp'].values.reshape(-1, 1)
            y = df_casa['geracao_kwh'].values
            model = LinearRegression().fit(X, y)
            ultima_hora = df_casa['hora_timestamp'].max()
            horas_futuras = np.array([ultima_hora + 3600 * i for i in range(1, horas_a_frente + 1)]).reshape(-1, 1)
            predicoes = model.predict(horas_futuras)
            geracao_prevista = predicoes.mean()
            previsoes.append({
                "id_casa": casa,
                "geracao_prevista": geracao_prevista,
                "predicoes_horas": predicoes,
                "horas_futuras": horas_futuras.flatten()
            })
        return previsoes

    # Executa previsões
    previsoes = prever_consumo_por_casa(df)
    previsoes_geracao = prever_geracao_por_casa(df)

    # Montar DataFrames de previsões **filtrando None** para evitar KeyError no merge
    
    df_previsoes_consumo = pd.DataFrame([
        {"id_casa": p["id_casa"], "Consumo Previsto (média 3h)": p["consumo_previsto"]}
        for p in previsoes if p["consumo_previsto"] is not None
    ])
    
    df_previsoes_geracao = pd.DataFrame([
        {"id_casa": p["id_casa"], "Geração Prevista (média 3h)": p["geracao_prevista"]}
        for p in previsoes_geracao if p["geracao_prevista"] is not None
    ])

    df_latest = df_latest.copy()
    df_latest = df_latest.merge(df_previsoes_consumo, on="id_casa", how="left")
    df_latest = df_latest.merge(df_previsoes_geracao, on="id_casa", how="left")

    # Exibe tabela geral
    st.subheader("🔢 Consumo e Geração por Casa - Previsão para as Próximas Horas")
    st.dataframe(df_latest.style.format({
        "Consumo Previsto (média 3h)": lambda x: f"{x:.2f}" if pd.notnull(x) else "—",
        "Geração Prevista (média 3h)": lambda x: f"{x:.2f}" if pd.notnull(x) else "—"
    }))

    # Seleção dinâmica da casa
    casas_disponiveis = sorted(df['id_casa'].unique())
    casa_selecionada = st.selectbox("🏠 Selecione a casa para ver o histórico e previsão:", casas_disponiveis, key="casa_selectbox")

    # Busca previsão da casa selecionada
    p = next((p for p in previsoes if p["id_casa"] == casa_selecionada), None)
    p_gen = next((p for p in previsoes_geracao if p["id_casa"] == casa_selecionada), None)

    # Título da seção
    st.subheader(f"📈 Histórico e Previsão")

    # Validação e exibição de gráfico
    if p is None or p["consumo_previsto"] is None:
        st.warning(f"⚠️ Casa {casa_selecionada}: dados insuficientes para previsão.")
    else:
        df_casa = df[df['id_casa'] == casa_selecionada].sort_values('timestamp')

        fig = go.Figure()

        # Consumo histórico
        fig.add_trace(go.Scatter(
            x=df_casa['timestamp'],
            y=df_casa['consumo_kwh'],
            mode='lines+markers',
            name='Consumo Histórico'
        ))

        # Previsão de consumo
        fig.add_trace(go.Scatter(
            x=[datetime.fromtimestamp(ts) for ts in p["horas_futuras"]],
            y=p["predicoes_horas"],
            mode='lines+markers',
            name='Previsão de Consumo',
            line=dict(dash='dash', color='red')
        ))

        # Previsão de geração (se disponível)
        if p_gen and p_gen["geracao_prevista"] is not None:
            fig.add_trace(go.Scatter(
                x=[datetime.fromtimestamp(ts) for ts in p_gen["horas_futuras"]],
                y=p_gen["predicoes_horas"],
                mode='lines+markers',
                name='Previsão de Geração',
                line=dict(dash='dot', color='green')
            ))

        # Layout final do gráfico
        fig.update_layout(
            title=f"📊 Histórico e Previsão - Casa {casa_selecionada}",
            xaxis_title="Data/Hora",
            yaxis_title="kWh",
            height=400,
            margin=dict(l=20, r=20, t=50, b=30)
        )

        st.plotly_chart(fig, use_container_width=True)

    # CLUSTERIZAÇÃO
    st.subheader("🔍 Perfil de Consumo por Cômodo (Cluster)")
    ultima_leitura_comodos = df_comodos.sort_values("timestamp").groupby("id_casa").tail(1)
    kmeans = KMeans(n_clusters=2, n_init=10).fit(ultima_leitura_comodos.iloc[:, 2:])
    ultima_leitura_comodos["cluster"] = kmeans.labels_

    radar_data = ultima_leitura_comodos.groupby("cluster")[["Quarto1", "Quarto2", "Sala", "Cozinha", "Piscina"]].mean()
    categorias = radar_data.columns.tolist()
    fig_radar = go.Figure()
    for i, row in radar_data.iterrows():
        fig_radar.add_trace(go.Scatterpolar(r=row, theta=categorias, fill='toself', name=f'Cluster {i}'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    # MAPA INTERATIVO COM TOOLTIP
    st.subheader("📍 Casas no Mapa por Cluster")

    cluster_por_casa = ultima_leitura_comodos.set_index("id_casa")["cluster"]
    casas_df = pd.DataFrame(casas_coords)
    casas_df["cluster"] = casas_df["id"].map(cluster_por_casa)
    cluster_colors = {
        0: [0, 200, 255, 180],
        1: [255, 100, 100, 180],
    }
    casas_df["color"] = casas_df["cluster"].map(cluster_colors).apply(lambda x: x if isinstance(x, list) else [150, 150, 150, 160])

    def get_comodo_mais_ativo(row):
        leitura = ultima_leitura_comodos[ultima_leitura_comodos["id_casa"] == row["id"]]
        if leitura.empty:
            return "Sem dados"
        leitura = leitura.iloc[0]
        comodos = ["Quarto1", "Quarto2", "Sala", "Cozinha", "Piscina"]
        valores = leitura[comodos].values
        idx_max = valores.argmax()
        comodo_nome = comodos[idx_max]
        ativacoes = valores[idx_max]
        return f"{comodo_nome} ({int(ativacoes)}x)"

    casas_df["comodo_ativo"] = casas_df.apply(get_comodo_mais_ativo, axis=1)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=casas_df,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius=60,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": """
            <b>Casa:</b> {id} <br/>
            <b>Cluster:</b> {cluster} <br/>
            <b>Cômodo mais ativo:</b> {comodo_ativo}
        """,
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "fontWeight": "bold",
            "padding": "10px",
            "borderRadius": "5px",
        }
    }

    view_state = pdk.ViewState(latitude=-28.482, longitude=-48.781, zoom=13)

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
    )

    st.pydeck_chart(deck)

    # ALERTAS
    st.subheader("🚨 Alertas de Uso Intenso por Cômodo (Baseado em Cluster)")

    def icone_alerta(valor, media, desvio):
        if valor > media + 2 * desvio:
            return "🔴"  # Muito acima do normal
        elif valor > media + desvio:
            return "🟠"  # Acima do esperado
        elif valor > media:
            return "🟡"  # Levemente acima da média
        return None  # Sem alerta

    for cluster_id in sorted(ultima_leitura_comodos["cluster"].unique()):
        st.markdown(f"### 🔹 Cluster {cluster_id}")

        cluster_dados = ultima_leitura_comodos[ultima_leitura_comodos["cluster"] == cluster_id]

        for comodo in ["Quarto1", "Quarto2", "Sala", "Cozinha", "Piscina"]:
            media = cluster_dados[comodo].mean()
            desvio = cluster_dados[comodo].std()

            for _, row in cluster_dados.iterrows():
                ativacoes = row[comodo]
                alerta = icone_alerta(ativacoes, media, desvio)
                if alerta:
                    intensidade = (
                        "uso **excessivo**" if alerta == "🔴"
                        else "uso **acima do esperado**" if alerta == "🟠"
                        else "uso **elevado**"
                    )
                    st.markdown(
                        f"{alerta} Casa {int(row['id_casa'])} no Cluster {cluster_id}: "
                        f"{intensidade} no **{comodo}** ({int(ativacoes)} ativações | média: {media:.1f})"
                    )

    # EXPORTAR
    if st.button("📄 Exportar PDF"):
        exportar_pdf(df_latest)
        st.success("PDF gerado com sucesso!")

# IMPORTAÇÃO DO AGENTE LLM
from agente_llm import app_agente

# MENU LATERAL
menu = st.sidebar.radio("Menu", ["Simulador Solar", "Agente Inteligente"])

if menu == "Simulador Solar":
    executar_simulador_solar()
elif menu == "Agente Inteligente":
    app_agente()

      
      


# Obs: No futuro podemos integrar dados reais de consumo e geração com sensores físicos ou APIs externas.
