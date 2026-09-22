import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import preparo_dados

# Injetar a classe customizada no __main__ para desserialização do joblib
sys.modules['__main__'].PreparadorDadosTransformer = preparo_dados.PreparadorDadosTransformer

st.set_page_config(page_title="Análise de Score de Crédito", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('modelo_xgb_credito.joblib')

@st.cache_data
def load_raw_csv():
    # Lê o dataset bruto de referência enviado para o repositório
    df = pd.read_csv('credito_tratado.csv')
    cols_alvo = [c for c in ['inadimplente', 'target', 'id', 'ID', 'Unnamed: 0'] if c in df.columns]
    return df.drop(columns=cols_alvo)

pipeline = load_model()
df_raw = load_raw_csv()

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
    # 1. Identificar a lista exata de colunas esperadas pelo estimador/imputer do pipeline
    primeiro_passo = pipeline.steps[0][1] if hasattr(pipeline, 'steps') else None
    
    if primeiro_passo and hasattr(primeiro_passo, 'feature_names_in_'):
        colunas_exatas = list(primeiro_passo.feature_names_in_)
    elif hasattr(pipeline, 'feature_names_in_'):
        colunas_exatas = list(pipeline.feature_names_in_)
    else:
        colunas_exatas = list(df_raw.columns)

    # 2. Reconstruir o DataFrame garantindo as colunas e tipos exatos esperados
    dados_dict = {}
    for col in colunas_exatas:
        c_lower = col.lower()
        if c_lower == 'idade':
            dados_dict[col] = float(idade)
        elif 'renda' in c_lower:
            dados_dict[col] = float(renda_mensal)
        elif 'linha' in c_lower or 'credito' in c_lower:
            dados_dict[col] = float(num_linhas_credito)
        elif 'depend' in c_lower:
            dados_dict[col] = float(dependentes)
        elif 'restrin' in c_lower:
            dados_dict[col] = 1.0 if restringido == "Sim" else 0.0
        elif 'inadimpl' in c_lower or 'historico' in c_lower:
            dados_dict[col] = 1.0 if historico_inadimplencia == "Sim" else 0.0
        else:
            # Caso o modelo tenha colunas secundárias, preenche com a média do CSV
            dados_dict[col] = float(df_raw[col].dropna().mean()) if col in df_raw.columns else 0.0

    # Criar o DataFrame na ordem exata exigida pelo scikit-learn
    dados_cliente = pd.DataFrame([dados_dict])[colunas_exatas]

    # Previsão sem falhas de validação
    prob = pipeline.predict_proba(dados_cliente)[0][1]
    
    st.subheader("Resultado da Análise:")
    st.metric(label="Risco de Inadimplência", value=f"{prob * 100:.2f}%")
    
    if prob > 0.5:
        st.error("⚠️ Alto Risco de Crédito!")
    else:
        st.success("✅ Baixo Risco de Crédito (Aprovado)!")
