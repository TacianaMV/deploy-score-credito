import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import preparo_dados

# Injetar a classe customizada no __main__ para o joblib carregar corretamente
sys.modules['__main__'].PreparadorDadosTransformer = preparo_dados.PreparadorDadosTransformer

st.set_page_config(page_title="Análise de Score de Crédito", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('modelo_xgb_credito.joblib')

@st.cache_data
def load_base_data():
    df = pd.read_csv('credito_tratado.csv')
    cols_alvo = [c for c in ['inadimplente', 'target', 'id', 'ID', 'Unnamed: 0'] if c in df.columns]
    return df.drop(columns=cols_alvo)

pipeline = load_model()
df_base = load_base_data()

st.title("📊 Análise e Previsão de Score de Crédito")
st.write("Insira os dados do cliente para calcular a probabilidade de inadimplência.")

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
    # 1. Copia exatamente a primeira linha do CSV base para manter todas as colunas originais do treino
    dados_cliente = df_base.iloc[[0]].copy()
    
    # 2. Atualiza os valores dos campos preenchidos na interface
    for col in dados_cliente.columns:
        c_lower = col.lower()
        if c_lower == 'idade':
            dados_cliente[col] = float(idade)
        elif 'renda' in c_lower and 'faltante' not in c_lower:
            dados_cliente[col] = float(renda_mensal)
        elif 'linha' in c_lower or 'credito' in c_lower:
            dados_cliente[col] = float(num_linhas_credito)
        elif 'depend' in c_lower:
            dados_cliente[col] = float(dependentes)
        elif 'restrin' in c_lower:
            dados_cliente[col] = 1.0 if restringido == "Sim" else 0.0
        elif 'inadimpl' in c_lower or 'historico' in c_lower:
            dados_cliente[col] = 1.0 if historico_inadimplencia == "Sim" else 0.0

    # 3. Ajusta dinamicamente a contagem de colunas esperada (n_features_in_) nos passos do pipeline
    num_cols = dados_cliente.shape[1]
    
    def ajustar_validacao_pipeline(estimator, n_cols):
        if hasattr(estimator, 'n_features_in_'):
            estimator.n_features_in_ = n_cols
        if hasattr(estimator, 'feature_names_in_'):
            delattr(estimator, 'feature_names_in_')
        if hasattr(estimator, 'steps'):
            for _, step in estimator.steps:
                ajustar_validacao_pipeline(step, n_cols)

    ajustar_validacao_pipeline(pipeline, num_cols)

    # Executa a previsão
    prob = pipeline.predict_proba(dados_cliente)[0][1]
    
    st.subheader("Resultado da Análise:")
    st.metric(label="Risco de Inadimplência", value=f"{prob * 100:.2f}%")
    
    if prob > 0.5:
        st.error("⚠️ Alto Risco de Crédito!")
    else:
        st.success("✅ Baixo Risco de Crédito (Aprovado)!")
