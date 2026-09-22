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
    # Lê o CSV de referência
    df = pd.read_csv('credito_tratado.csv')
    # Remove a coluna alvo/target e colunas de id se existirem
    cols_para_remover = [c for c in ['inadimplente', 'target', 'id', 'ID', 'Unnamed: 0'] if c in df.columns]
    df_features = df.drop(columns=cols_para_remover)
    return df_features

pipeline = load_model()
df_sample = load_sample_data()

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
    # Criar um DataFrame com 1 linha contendo EXATAMENTE todas as colunas do dataset original
    dados_cliente = df_sample.iloc[[0]].copy()
    
    # Sobra/Preenche os campos informados pelo formulário se a coluna existir
    for col in dados_cliente.columns:
        if col == 'idade':
            dados_cliente[col] = idade
        elif col in ['renda_mensal', 'renda']:
            dados_cliente[col] = renda_mensal
        elif col in ['num_linhas_credito', 'linhas_credito']:
            dados_cliente[col] = num_linhas_credito
        elif col in ['dependentes', 'num_dependentes']:
            dados_cliente[col] = dependentes
        elif col == 'restringido':
            dados_cliente[col] = 1 if restringido == "Sim" else 0
        elif col in ['historico_inadimplencia', 'inadimplente_anterior']:
            dados_cliente[col] = 1 if historico_inadimplencia == "Sim" else 0

    # Garantir a ordem exata das colunas
    dados_cliente = dados_cliente[df_sample.columns]

    # Previsão
    prob = pipeline.predict_proba(dados_cliente)[0][1]
    
    st.subheader("Resultado da Análise:")
    st.metric(label="Risco de Inadimplência", value=f"{prob * 100:.2f}%")
    
    if prob > 0.5:
        st.error("⚠️ Alto Risco de Crédito!")
    else:
        st.success("✅ Baixo Risco de Crédito (Aprovado)!")
