import os
import sqlite3
import streamlit as st
import openai

# Configurar API Key do OpenAI
openai.api_key = "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"

# Caminho do banco de dados
DB_PATH = "smartcity_laguna.db"

# Função que envia a pergunta para o modelo GPT e retorna resposta
def perguntar_agente(pergunta):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tabelas = cursor.fetchall()

        contexto = f"O banco possui as seguintes tabelas: {[t[0] for t in tabelas]}.\nPergunta: {pergunta}"

        resposta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "Você é um assistente inteligente de energia que responde perguntas com base nos dados de um banco SQLite da cidade inteligente."},
                {"role": "user", "content": contexto}
            ]
        )
        return resposta.choices[0].message.content
    except Exception as e:
        return f"Erro: {str(e)}"

# Função principal da interface Streamlit
def app_agente():
    st.title("🤖 EnergiA - Seu Assistente de Energia")
    st.markdown(
        """
        **Pergunte sobre consumo, geração, sensores e alertas no sistema de energia da Smart City Laguna.**
        
        Exemplos:
        - Qual casa mais gerou energia?
        - Qual casa teve maior consumo?
        - Quais casas estão com excedente positivo?
        """
    )

    pergunta = st.text_input("💬 Faça uma pergunta personalizada:")
    resposta_area = st.empty()

    if pergunta:
        with st.spinner("Consultando base de dados..."):
            resposta = perguntar_agente(pergunta)
            if resposta.startswith("Erro:"):
                st.error(resposta)
            else:
                resposta_area.markdown(f"**🤖 Resposta:** {resposta}")

    st.markdown("---")
    st.write("Ou selecione uma ação rápida:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🛰️ Monitorar sensores"):
            with st.spinner("Consultando sensores..."):
                resposta = perguntar_agente("Mostre os dados dos sensores.")
                if resposta.startswith("Erro:"):
                    st.error(resposta)
                else:
                    resposta_area.markdown(f"**🤖 Resposta:** {resposta}")

        if st.button("💡 Dicas de economia"):
            with st.spinner("Buscando dicas..."):
                resposta = perguntar_agente("Me dê dicas para economizar energia.")
                if resposta.startswith("Erro:"):
                    st.error(resposta)
                else:
                    resposta_area.markdown(f"**🤖 Resposta:** {resposta}")

    with col2:
        if st.button("📊 Consumo hoje"):
            with st.spinner("Consultando consumo do dia..."):
                resposta = perguntar_agente("Qual foi o consumo de energia hoje?")
                if resposta.startswith("Erro:"):
                    st.error(resposta)
                else:
                    resposta_area.markdown(f"**🤖 Resposta:** {resposta}")

        if st.button("📈 Previsão de consumo"):
            with st.spinner("Gerando previsão..."):
                resposta = perguntar_agente("Faça uma previsão do consumo futuro.")
                if resposta.startswith("Erro:"):
                    st.error(resposta)
                else:
                    resposta_area.markdown(f"**🤖 Resposta:** {resposta}")

    with col3:
        if st.button("💰 Insights de gastos"):
            with st.spinner("Analisando gastos..."):
                resposta = perguntar_agente("Quais os insights sobre os gastos de energia?")
                if resposta.startswith("Erro:"):
                    st.error(resposta)
                else:
                    resposta_area.markdown(f"**🤖 Resposta:** {resposta}")

        if st.button("📄 Gerar relatório"):
            with st.spinner("Gerando relatório..."):
                resposta = perguntar_agente("Gere um relatório de consumo e geração.")
                if resposta.startswith("Erro:"):
                    st.error(resposta)
                else:
                    resposta_area.markdown(f"**🤖 Resposta:** {resposta}")
