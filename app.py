import joblib
import pandas as pd
import numpy as np

# Load the pickle/joblib pipeline file
# Source 3 contains the serialized pipeline
with open('preparo_dados.py', 'r') as f:
    pass # source 2 is preparo_dados.py, source 3 is the binary model/pipeline

# Let's inspect source 3 file using joblib / pickle directly
# Notice that source 3 is saved under the name where python can load it, let's write source 3 binary content or load it if it's in the workspace.
# Let's list files in current directory to locate the model file name.
import os
print("Files in workspace:", os.listdir('.'))
# Import custom class so joblib can deserialize it
from preparo_dados import PreparadorDadosTransformer

pipeline = joblib.load('modelo_xgb_credito.joblib')
print("Pipeline loaded successfully:")
print(pipeline)

# Load data and run pipeline to inspect output
df = pd.read_csv('credito_tratado.csv')
print("\nDataset columns:", df.columns.tolist())
print("Dataset shape:", df.shape)

X_transformed = pipeline.named_steps['preparo'].transform(df)
print("\nTransformed columns:", X_transformed.columns.tolist())
print(X_transformed.head(2))
import sys
import preparo_dados

# Inject PreparadorDadosTransformer into __main__ so pickle can resolve it
sys.modules['__main__'].PreparadorDadosTransformer = preparo_dados.PreparadorDadosTransformer

pipeline = joblib.load('modelo_xgb_credito.joblib')
print("Pipeline loaded successfully:", pipeline)

df = pd.read_csv('credito_tratado.csv')
print("\nFirst row sample prediction:")
X_test = df.head(5)
if 'inadimplente' in X_test.columns:
    X_test = X_test.drop(columns=['inadimplente'])

preds = pipeline.predict_proba(X_test)
print(preds)
# Let's inspect source files and details directly without xgboost
with open('preparo_dados.py', 'r') as f:
    code = f.read()

print("Code in preparo_dados.py:\n")
print(code)
# Check data summary
df = pd.read_csv('credito_tratado.csv')
print("Dataset summary:")
print(df.info())
print("\nFirst 3 rows:")
print(df.head(3))