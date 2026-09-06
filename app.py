import streamlit as st 
import pandas as pd 
import joblib
st.title("Manutenção Preditiva - RUL de Motores")
st.write("Carrega dados de sensores de um motor para prever o tempo de vida útil restante")
model = joblib.load("model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
uploaded = st.file_uploader("Carrega um CSV com dados do motor", type = ["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    preds = model.predict(df[feature_cols])
    st.line_chart(preds)
    st.metric("RUL previsto (último ciclo)", f"{preds[-1]:.0f} ciclos")
    if preds[-1]<20:
        st.warning("RUL baixo - agendar manutenção")
