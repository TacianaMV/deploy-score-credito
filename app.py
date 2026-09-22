import streamlit as st
import pandas as pd
import joblib
import sys
import preparo_dados

# Injetar a classe customizada no __main__
sys.modules['__main__'].PreparadorDadosTransformer = preparo_dados.PreparadorDadosTransformer

st.set_page_config(page_title="Análise de Score de Crédito", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('modelo_xgb_credito.joblib')

@st.cache_data
def load_sample_data():
    return pd.read_csv('credito_tratado.csv')

pipeline = load_model()
df_sample = load_sample_data()

st.title("📊 Análise e Previsão de Score de Crédito")
st.write("Insira os dados do cliente para calcular a probabilidade de inadimplência.")

# Criar uma cópia da primeira linha do dataset para garantir a presença de todas as colunas esperadas
dados_cliente = df_sample.drop(columns=['inadimplente'], errors='ignore').iloc[[0]].copy()

# Formulário para entrada de dados
with st.form("form_credito"):
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.number_input("Idade", min_value=18, max_value=100, value=35)
        renda_mensal = st.number_input("Renda Mensal (R$)", min_value=0.0, value=5000.0)
        num_linhas_credito = st.number_input("Número de Linhas de Crédito", min_value=0, value=3)
        
    with col2:
        dependentes = st.number_input("Número de Dependentes", min_value=0, max_value=20, value=0)
        restringido = st.selectbox("Possui Restrição Nome?", ["Não", "Sim"])
        historico_inadimplencia = st.selectbox("Histórico Anterior de Inadimplência?", ["Não", "Sim"])
        
    btn_predict = st.form_submit_button("Calcular Score")

if btn_predict:
    # Atualizar as colunas editadas pelo utilizador no DataFrame com a estrutura completa
    if 'idade' in dados_cliente.columns:
        dados_cliente['idade'] = idade
    if 'renda_mensal' in dados_cliente.columns:
        dados_cliente['renda_mensal'] = renda_mensal
    if 'num_linhas_credito' in dados_cliente.columns:
        dados_cliente['num_linhas_credito'] = num_linhas_credito
    if 'dependentes' in dados_cliente.columns:
        dados_cliente['dependentes'] = dependentes
    if 'restringido' in dados_cliente.columns:
        dados_cliente['restringido'] = 1 if restringido == "Sim" else 0
    if 'historico_inadimplencia' in dados_cliente.columns:
        dados_cliente['historico_inadimplencia'] = 1 if historico_inadimplencia == "Sim" else 0
    
    # Previsão
    prob = pipeline.predict_proba(dados_cliente)[0][1]
    
    st.subheader("Resultado da Análise:")
    st.metric(label="Risco de Inadimplência", value=f"{prob * 100:.2f}%")
    
    if prob > 0.5:
        st.error("⚠️ Alto Risco de Crédito!")
    else:
        st.success("✅ Baixo Risco de Crédito (Aprovado)!")
