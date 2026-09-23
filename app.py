import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import preparo_dados

# Injetar a classe customizada para desserialização do joblib
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
    # 1. Tentar obter os nomes exatos das colunas do modelo
    first_step = pipeline.steps[0][1] if hasattr(pipeline, 'steps') else pipeline
    cols_esperadas = None

    if hasattr(first_step, 'feature_names_in_'):
        cols_esperadas = list(first_step.feature_names_in_)
    elif hasattr(pipeline, 'feature_names_in_'):
        cols_esperadas = list(pipeline.feature_names_in_)

    # 2. Se não houver feature_names_in_, verificar a quantidade n_features_in_
    n_expected = getattr(first_step, 'n_features_in_', getattr(pipeline, 'n_features_in_', None))

    if cols_esperadas:
        # Reconstruir dicionário com os nomes das colunas esperadas
        dados_dict = {}
        for col in cols_esperadas:
            if col in df_base.columns:
                dados_dict[col] = float(df_base[col].dropna().iloc[0]) if not df_base[col].dropna().empty else 0.0
            else:
                dados_dict[col] = 0.0
        dados_cliente = pd.DataFrame([dados_dict])[cols_esperadas]
    else:
        # Se usamos o df_base
        dados_cliente = df_base.iloc[[0]].copy()
        # Se a quantidade esperada for maior que o número de colunas do df_base, preencher com colunas dummy
        if n_expected and dados_cliente.shape[1] < n_expected:
            massa_faltante = n_expected - dados_cliente.shape[1]
            for i in range(massa_faltante):
                dados_cliente[f'col_aux_{i}'] = 0.0

    # 3. Atualizar os valores informados pelo usuário
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

    # Executar a previsão
    prob = pipeline.predict_proba(dados_cliente)[0][1]
    
    st.subheader("Resultado da Análise:")
    st.metric(label="Risco de Inadimplência", value=f"{prob * 100:.2f}%")
    
    if prob > 0.5:
        st.error("⚠️ Alto Risco de Crédito!")
    else:
        st.success("✅ Baixo Risco de Crédito (Aprovado)!")
