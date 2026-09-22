import streamlit as st
import pandas as pd
import joblib
import sys
import preparo_dados

# Injetar a classe customizada no __main__ para o joblib desmaterializar corretamente
sys.modules['__main__'].PreparadorDadosTransformer = preparo_dados.PreparadorDadosTransformer

st.set_page_config(page_title="Análise de Score de Crédito", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('modelo_xgb_credito.joblib')

pipeline = load_model()

st.title("📊 Análise e Previsão de Score de Crédito")
st.write("Insira os dados do cliente para calcular a probabilidade de inadimplência.")

# Descobrir automaticamente os nomes das colunas esperadas pelo pipeline
try:
    # Tenta extrair diretamente as feature_names_in_ do primeiro passo do pipeline
    primeiro_passo = pipeline.steps[0][1]
    if hasattr(primeiro_passo, 'feature_names_in_'):
        colunas_esperadas = list(primeiro_passo.feature_names_in_)
    elif hasattr(pipeline, 'feature_names_in_'):
        colunas_esperadas = list(pipeline.feature_names_in_)
    else:
        # Se não encontrar no pipeline, lê as colunas do CSV descartando o alvo
        df_csv = pd.read_csv('credito_tratado.csv')
        cols_alvo = ['inadimplente', 'target', 'id', 'ID', 'Unnamed: 0']
        colunas_esperadas = [c for c in df_csv.columns if c not in cols_alvo]
except Exception:
    df_csv = pd.read_csv('credito_tratado.csv')
    cols_alvo = ['inadimplente', 'target', 'id', 'ID', 'Unnamed: 0']
    colunas_esperadas = [c for c in df_csv.columns if c not in cols_alvo]

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
    # Criar um dicionário inicializado com valores predefinidos/nulos para todas as colunas
    dados_dict = {col: [0] for col in colunas_esperadas}
    
    # Preencher as colunas mapeadas do formulário
    for col in colunas_esperadas:
        col_lower = col.lower()
        if col_lower == 'idade':
            dados_dict[col] = [idade]
        elif 'renda' in col_lower:
            dados_dict[col] = [renda_mensal]
        elif 'linha' in col_lower or 'credito' in col_lower:
            dados_dict[col] = [num_linhas_credito]
        elif 'depend' in col_lower:
            dados_dict[col] = [dependentes]
        elif 'restrin' in col_lower:
            dados_dict[col] = [1 if restringido == "Sim" else 0]
        elif 'inadimpl' in col_lower or 'historico' in col_lower:
            dados_dict[col] = [1 if historico_inadimplencia == "Sim" else 0]

    # Montar o DataFrame respeitando estritamente a ordem exata das colunas esperadas
    dados_cliente = pd.DataFrame(dados_dict)[colunas_esperadas]
    
    # Previsão
    prob = pipeline.predict_proba(dados_cliente)[0][1]
    
    st.subheader("Resultado da Análise:")
    st.metric(label="Risco de Inadimplência", value=f"{prob * 100:.2f}%")
    
    if prob > 0.5:
        st.error("⚠️ Alto Risco de Crédito!")
    else:
        st.success("✅ Baixo Risco de Crédito (Aprovado)!")
