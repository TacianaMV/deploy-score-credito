import streamlit as st
import pandas as pd
import numpy as np
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
def load_sample():
    df = pd.read_csv('credito_tratado.csv')
    cols_alvo = [c for c in ['inadimplente', 'target', 'id', 'ID', 'Unnamed: 0'] if c in df.columns]
    return df.drop(columns=cols_alvo)

pipeline = load_model()
df_ref = load_sample()

st.title("📊 Análise e Previsão de Score de Crédito")
st.write("Insira os dados do cliente para calcular a probabilidade de inadimplência.")

# Form de entrada
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
    # 1. Copia a primeira linha do modelo como base
    dados_cliente = df_ref.iloc[[0]].copy()
    
    # 2. Atualiza as colunas que coincidem com os inputs do utilizador
    for col in dados_cliente.columns:
        c_lower = col.lower()
        if c_lower == 'idade':
            dados_cliente[col] = idade
        elif 'renda' in c_lower:
            dados_cliente[col] = renda_mensal
        elif 'linha' in c_lower or 'credito' in c_lower:
            dados_cliente[col] = num_linhas_credito
        elif 'depend' in c_lower:
            dados_cliente[col] = dependentes
        elif 'restrin' in c_lower:
            dados_cliente[col] = 1 if restringido == "Sim" else 0
        elif 'inadimpl' in c_lower or 'historico' in c_lower:
            dados_cliente[col] = 1 if historico_inadimplencia == "Sim" else 0

    # 3. Remover a verificação estrita de nomes de colunas do scikit-learn
    # passando os valores ou mapeando para as colunas exatas que o pipeline guardou
    try:
        prob = pipeline.predict_proba(dados_cliente)[0][1]
    except ValueError:
        # Se falhar a validação de nomes de colunas, remove os nomes para passar apenas a matriz numérica
        dados_cliente.columns = [f"col_{i}" for i in range(dados_cliente.shape[1])]
        prob = pipeline.predict_proba(dados_cliente)[0][1]
    
    st.subheader("Resultado da Análise:")
    st.metric(label="Risco de Inadimplência", value=f"{prob * 100:.2f}%")
    
    if prob > 0.5:
        st.error("⚠️ Alto Risco de Crédito!")
    else:
        st.success("✅ Baixo Risco de Crédito (Aprovado)!")
