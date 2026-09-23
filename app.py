import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import preparo_dados

# Injetar a classe customizada no __main__ para desserialização correta no joblib
sys.modules['__main__'].PreparadorDadosTransformer = preparo_dados.PreparadorDadosTransformer

st.set_page_config(page_title="Análise de Score de Crédito", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('modelo_xgb_credito.joblib')

@st.cache_data
def load_base_data():
    return pd.read_csv('credito_tratado.csv')

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
    # 1. Copia a linha de referência da base
    dados_cliente = df_base.iloc[[0]].copy()

    # 2. Preenche os campos fornecidos pelo utilizador
    for col in dados_cliente.columns:
        c_lower = str(col).lower()
        if c_lower == 'idade':
            dados_cliente[col] = float(idade)
        elif 'renda' in c_lower and 'faltante' not in c_lower:
            dados_cliente[col] = float(renda_mensal)
        elif 'linha' in c_lower or 'credito' in c_lower:
            dados_cliente[col] = float(num_linhas_credito)
        elif 'depend' in c_lower and 'ausente' not in c_lower:
            dados_cliente[col] = float(dependentes)
        elif 'restrin' in c_lower:
            dados_cliente[col] = 1.0 if restringido == "Sim" else 0.0
        elif 'inadimpl' in c_lower or 'historico' in c_lower:
            dados_cliente[col] = 1.0 if historico_inadimplencia == "Sim" else 0.0

    # 3. Ajuste dinâmico do número de colunas exigido pelo primeiro passo do pipeline
    step_0 = pipeline.steps[0][1] if hasattr(pipeline, 'steps') else pipeline

    if hasattr(step_0, 'feature_names_in_'):
        cols_req = list(step_0.feature_names_in_)
        for c in cols_req:
            if c not in dados_cliente.columns:
                dados_cliente[c] = 0.0
        dados_cliente = dados_cliente[cols_req]
    elif hasattr(step_0, 'n_features_in_'):
        n_req = step_0.n_features_in_
        if dados_cliente.shape[1] < n_req:
            for i in range(n_req - dados_cliente.shape[1]):
                dados_cliente[f'col_pad_{i}'] = 0.0
        elif dados_cliente.shape[1] > n_req:
            dados_cliente = dados_cliente.iloc[:, :n_req]

    # 4. Ajuste para o SimpleImputer (caso seja o segundo passo do pipeline)
    if hasattr(pipeline, 'steps') and len(pipeline.steps) > 1:
        step_1 = pipeline.steps[1][1]
        if hasattr(step_1, 'n_features_in_'):
            try:
                X_test = step_0.transform(dados_cliente.copy())
                n_out = X_test.shape[1]
                n_req_1 = step_1.n_features_in_
                if n_out < n_req_1:
                    diff = n_req_1 - n_out
                    for i in range(diff):
                        dados_cliente[f'feature_extra_{i}'] = 0.0
            except Exception:
                pass

    # Executa a previsão
    prob = pipeline.predict_proba(dados_cliente)[0][1]
    
    st.subheader("Resultado da Análise:")
    st.metric(label="Risco de Inadimplência", value=f"{prob * 100:.2f}%")
    
    if prob > 0.5:
        st.error("⚠️ Alto Risco de Crédito!")
    else:
        st.success("✅ Baixo Risco de Crédito (Aprovado)!")
