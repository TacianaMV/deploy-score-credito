import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PreparadorDadosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mediana_renda_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if 'renda_mensal' in X_df.columns:
            self.mediana_renda_ = X_df['renda_mensal'].median()
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Criação de flags e imputação
        df['renda_faltante'] = df['renda_mensal'].isnull().astype(int)
        df['dependentes_ausentes'] = df['dependentes'].isnull().astype(int)
        df['dependentes'] = df['dependentes'].fillna(0)
        
        mediana = self.mediana_renda_ if self.mediana_renda_ is not None else df['renda_mensal'].median()
        df['renda_mensal'] = df['renda_mensal'].fillna(mediana)
        
        # Tratamento de outliers e flags de atraso
        colunas_atraso = ['atrasos_30_59_dias', 'atrasos_60_89_dias', 'atrasos_90_mais_dias']
        df['flag_atraso_extremo'] = ((df['atrasos_30_59_dias'] >= 96) | 
                                     (df['atrasos_60_89_dias'] >= 96) | 
                                     (df['atrasos_90_mais_dias'] >= 96)).astype(int)
        
        for col in colunas_atraso:
            df[col] = df[col].apply(lambda x: min(x, 20))
            
        df['renda_mensal'] = df['renda_mensal'].apply(lambda x: min(x, 50000.0))
        
        # Engenharia de Recursos (Feature Engineering)
        df['renda_por_dependente'] = df['renda_mensal'] / (df['dependentes'] + 1)
        df['sobra_caixa'] = df['renda_mensal'] * (1 - df['razao_divida'])
        
        self.feature_names_out_ = np.array(df.columns)
        return df
