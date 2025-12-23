import streamlit as st
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ===============================
# CONFIGURAÇÃO DA PÁGINA
# ===============================
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Sistema Preditivo de Obesidade")
st.markdown("""
Este sistema utiliza **Machine Learning** para auxiliar profissionais da saúde  
na **predição do nível de obesidade** com base em dados clínicos e comportamentais.
""")
st.markdown("---")
st.header("📊 Visão Analítica – Insights sobre Obesidade")

df_dashboard = pd.read_csv("data/Obesity.csv")

st.subheader("Distribuição dos níveis de obesidade")
st.bar_chart(df_dashboard["Obesity"].value_counts())

st.subheader("Atividade física vs Obesidade")
st.bar_chart(
    df_dashboard.groupby("Obesity")["FAF"].mean()
)

st.subheader("Consumo de água vs Obesidade")
st.bar_chart(
    df_dashboard.groupby("Obesity")["CH2O"].mean()
)

st.subheader("Histórico familiar vs Obesidade")
st.bar_chart(
    df_dashboard.groupby("Obesity")["family_history"].apply(lambda x: (x == "yes").mean())
)

st.markdown("""
### 🧠 Principais Insights:
- Pessoas com **menor frequência de atividade física** tendem a níveis mais elevados de obesidade.
- O **baixo consumo de água** está associado a maiores níveis de obesidade.
- O **histórico familiar** é um fator relevante e recorrente nos níveis mais altos.
- Há forte influência de **hábitos alimentares** no diagnóstico.
""")

# ===============================
# TREINAMENTO DO MODELO
# ===============================
@st.cache_resource
def train_model():
    df = pd.read_csv("data/Obesity.csv")

    cols_round = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for col in cols_round:
        df[col] = df[col].round().astype(int)

    X = df.drop("Obesity", axis=1)
    y = df["Obesity"]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    )

    model.fit(X, y)
    return model

model = train_model()

# ===============================
# FORMULÁRIO
# ===============================
st.header("📋 Informações do Paciente")

with st.form("form_paciente"):

    col1, col2 = st.columns(2)

    with col1:
        gender_pt = st.selectbox("Gênero", ["Masculino", "Feminino"])
gender = "Male" if gender_pt == "Masculino" else "Female"
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
    Este sistema é um **apoio à decisão clínica** e não substitui avaliação médica.
    """)
