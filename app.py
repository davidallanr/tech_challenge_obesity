# NÃO EXECUTE ANTES DE COLAR O CÓDIGO
import streamlit as st
import pandas as pd
import joblib

# ===============================
# CONFIGURAÇÃO DA PÁGINA
# ===============================
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Sistema Preditivo de Obesidade")
st.markdown("""
Este sistema utiliza **Machine Learning** para auxiliar profissionais da saúde  
na **predição do nível de obesidade** com base em dados clínicos e comportamentais.
""")

# ===============================
# CARREGAMENTO DO MODELO
# ===============================
def load_model():
    return joblib.load("model/modelo_obesidade.pkl")

model = load_model()

# ===============================
# FORMULÁRIO DE ENTRADA
# ===============================
st.header("📋 Informações do Paciente")

with st.form("form_paciente"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gênero", ["Male", "Female"])
        age = st.number_input("Idade", min_value=14, max_value=80, value=25)
        height = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70)
        weight = st.number_input("Peso (kg)", min_value=40.0, max_value=200.0, value=70.0)
        family_history = st.selectbox("Histórico familiar de sobrepeso?", ["yes", "no"])
        favc = st.selectbox("Consome alimentos altamente calóricos?", ["yes", "no"])
        smoke = st.selectbox("Fuma?", ["yes", "no"])

    with col2:
        fcvc = st.slider("Consumo de vegetais", 1, 3, 2)
        ncp = st.slider("Número de refeições diárias", 1, 4, 3)
        caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = st.slider("Consumo diário de água", 1, 3, 2)
        scc = st.selectbox("Monitora ingestão calórica?", ["yes", "no"])
        faf = st.slider("Frequência de atividade física", 0, 3, 1)
        tue = st.slider("Tempo em dispositivos eletrônicos", 0, 2, 1)
        calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox(
            "Meio de transporte",
            ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
        )

    submit = st.form_submit_button("🔍 Prever nível de obesidade")

# ===============================
# PREVISÃO
# ===============================
if submit:

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("📊 Resultado da Predição")
    st.success(f"Nível estimado de obesidade: **{prediction.replace('_', ' ')}**")

    st.markdown("""
    ⚠️ **Aviso:**  
    Este sistema é um **apoio à decisão clínica**, não substituindo avaliação médica.
    """)
