import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import random
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAÇÃO STREAMLIT ---
st.set_page_config(page_title="Simulador Solar", layout="wide")
st_autorefresh(interval=5000, key="refresh")

# --- BANCO DE DADOS ---
conn = sqlite3.connect("solar_data.db", check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS simulacoes (
    timestamp TEXT,
    id_casa INTEGER,
    consumo_kwh REAL,
    geracao_kwh REAL,
    excedente REAL
)''')
conn.commit()

# --- FUNÇÕES ---
def get_estacao(mes):
    if mes in [12, 1, 2]:
        return "verão"
    elif mes in [3, 4, 5]:
        return "outono"
    elif mes in [6, 7, 8]:
        return "inverno"
    else:
        return "primavera"

def simular_clima():
    return random.choices(
        population=["ensolarado", "parcialmente nublado", "nublado", "chuvoso"],
        weights=[0.4, 0.3, 0.2, 0.1],
        k=1
    )[0]

def fator_climatico(clima):
    return {
        "ensolarado": 1.0,
        "parcialmente nublado": 0.7,
        "nublado": 0.4,
        "chuvoso": 0.2
    }[clima]

# --- SIMULAÇÃO ---
st.title("🔆 Simulador de Energia Solar em Tempo Real")

potencia_kwp = st.slider("Potência do sistema (kWp)", 1.0, 10.0, 3.5, 0.5)
eficiencia = 0.8

agora = datetime.now()
hora_atual = agora.hour
mes_atual = agora.month
estacao = get_estacao(mes_atual)
clima = simular_clima()
fator = fator_climatico(clima)

# Horas de sol médias por estação
horas_sol_estacao = {
    "verão": random.uniform(5.5, 6.5),
    "outono": random.uniform(4.0, 5.5),
    "inverno": random.uniform(3.0, 4.5),
    "primavera": random.uniform(4.5, 6.0)
}
horas_sol = horas_sol_estacao[estacao]

# Exibe informações do ambiente simulado
st.markdown(f"""
🌤️ **Estação atual**: `{estacao.capitalize()}`
🕒 **Hora atual**: `{hora_atual}h`
🌦️ **Clima de hoje**: `{clima.capitalize()}`
☀️ **Horas de sol estimadas**: `{horas_sol:.2f} h`
🔧 **Fator de geração climática**: `{fator * 100:.0f}%`
""")

# Simular somente entre 6h e 18h
if 6 <= hora_atual <= 18:
    for id_casa in range(1, 11):
        consumo = random.uniform(8, 20)
        geracao = potencia_kwp * horas_sol * eficiencia * fator
        excedente = geracao - consumo
        timestamp = agora.strftime("%Y-%m-%d %H:%M:%S")

        c.execute("INSERT INTO simulacoes VALUES (?, ?, ?, ?, ?)", 
                  (timestamp, id_casa, consumo, geracao, excedente))
    conn.commit()
else:
    st.warning("⚠️ Fora do horário de captação solar (06h às 18h). Geração pausada.")

# --- VISUALIZAÇÃO ---
df = pd.read_sql_query("SELECT * FROM simulacoes", conn)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df_latest = df.sort_values("timestamp").groupby("id_casa").tail(1)

st.subheader("📊 Última Geração por Casa")
st.dataframe(df_latest)

st.line_chart(df_latest.set_index("id_casa")[["consumo_kwh", "geracao_kwh"]])

