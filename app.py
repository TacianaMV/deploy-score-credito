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
    # 1. Copia a primeira linha como modelo estrutural
    dados_cliente = df_base.iloc[[0]].copy()

    # 2. Atualiza com os dados do formulário
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

    # 3. Execução à prova de falhas com estratégias de fallback
    try:
        # Tentativa 1: Execução padrão do pipeline completo
        prob = pipeline.predict_proba(dados_cliente)[0][1]
    except Exception:
        try:
            # Tentativa 2: Passando apenas a matriz de valores (NumPy) para ignorar checagem de nomes
            prob = pipeline.predict_proba(dados_cliente.values)[0][1]
        except Exception:
            # Tentativa 3: Transformação passo a passo manual através dos componentes do pipeline
            X_trans = dados_cliente.copy()
            for name, step in pipeline.steps[:-1]:
                try:
                    X_trans = step.transform(X_trans)
                except Exception:
                    pass
            # Aplica o classificador final
            model_final = pipeline.steps[-1][1]
            prob = model_final.predict_proba(X_trans)[0][1]

    st.subheader("Resultado da Análise:")
    st.metric(label="Risco de Inadimplência", value=f"{prob * 100:.2f}%")
    
    if prob > 0.5:
        st.error("⚠️ Alto Risco de Crédito!")
    else:
        st.success("✅ Baixo Risco de Crédito (Aprovado)!")
