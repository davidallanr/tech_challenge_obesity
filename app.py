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
Este sistema utiliza **Aprendizado de Máquina (Machine Learning)** para auxiliar  
profissionais da saúde na **estimativa do nível de obesidade**, considerando dados  
clínicos, demográficos e comportamentais do paciente.
""")

# ===============================
# DASHBOARD ANALÍTICO
# ===============================
st.markdown("---")
st.header("📊 Visão Analítica – Insights sobre Obesidade")

df_dashboard = pd.read_csv("data/Obesity.csv")

st.subheader("Distribuição dos níveis de obesidade")
st.bar_chart(df_dashboard["Obesity"].value_counts())

st.subheader("Atividade física média por nível de obesidade")
st.bar_chart(df_dashboard.groupby("Obesity")["FAF"].mean())

st.subheader("Consumo médio de água por nível de obesidade")
st.bar_chart(df_dashboard.groupby("Obesity")["CH2O"].mean())

st.subheader("Proporção de histórico familiar de sobrepeso")
st.bar_chart(
    df_dashboard.groupby("Obesity")["family_history"]
    .apply(lambda x: (x == "yes").mean())
)

st.markdown("""
### 🧠 Principais Insights:
- Menor frequência de atividade física está associada a níveis mais elevados de obesidade.
- Baixo consumo de água aparece com maior frequência nos níveis mais altos.
- O histórico familiar de sobrepeso é um fator relevante nos diagnósticos.
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
st.markdown("---")
st.header("📋 Informações do Paciente")

with st.form("form_paciente"):

    col1, col2 = st.columns(2)

    with col1:
        genero_pt = st.selectbox("Gênero", ["Masculino", "Feminino"])
        gender = "Male" if genero_pt == "Masculino" else "Female"

        age = st.number_input("Idade", min_value=14, max_value=80, value=25)
        height = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70)
        weight = st.number_input("Peso (kg)", min_value=40.0, max_value=200.0, value=70.0)

        hist_fam_pt = st.selectbox("Histórico familiar de sobrepeso?", ["Sim", "Não"])
        family_history = "yes" if hist_fam_pt == "Sim" else "no"

        favc_pt = st.selectbox("Consome alimentos altamente calóricos?", ["Sim", "Não"])
        favc = "yes" if favc_pt == "Sim" else "no"

        smoke_pt = st.selectbox("Fuma?", ["Sim", "Não"])
        smoke = "yes" if smoke_pt == "Sim" else "no"

    with col2:
        fcvc = st.slider("Consumo de vegetais", 1, 3, 2)
        ncp = st.slider("Número de refeições diárias", 1, 4, 3)

        caec_pt = st.selectbox(
            "Costuma comer entre as refeições?",
            ["Não", "Às vezes", "Frequentemente", "Sempre"]
        )
        caec_map = {
            "Não": "no",
            "Às vezes": "Sometimes",
            "Frequentemente": "Frequently",
            "Sempre": "Always"
        }
        caec = caec_map[caec_pt]

        ch2o = st.slider("Consumo diário de água", 1, 3, 2)

        scc_pt = st.selectbox("Monitora a ingestão calórica diária?", ["Sim", "Não"])
        scc = "yes" if scc_pt == "Sim" else "no"

        faf = st.slider("Frequência de atividade física", 0, 3, 1)
        tue = st.slider("Tempo diário em dispositivos eletrônicos", 0, 2, 1)

        calc_pt = st.selectbox(
            "Consumo de bebidas alcoólicas",
            ["Não consome", "Às vezes", "Frequentemente", "Sempre"]
        )
        calc_map = {
            "Não consome": "no",
            "Às vezes": "Sometimes",
            "Frequentemente": "Frequently",
            "Sempre": "Always"
        }
        calc = calc_map[calc_pt]

        mtrans_pt = st.selectbox(
            "Meio de transporte utilizado",
            ["Carro", "Moto", "Bicicleta", "Transporte Público", "A pé"]
        )
        mtrans_map = {
            "Carro": "Automobile",
            "Moto": "Motorbike",
            "Bicicleta": "Bike",
            "Transporte Público": "Public_Transportation",
            "A pé": "Walking"
        }
        mtrans = mtrans_map[mtrans_pt]

    submit = st.form_submit_button("🔍 Prever nível de obesidade")

# ===============================
# RESULTADO
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

    label_map = {
        "Insufficient_Weight": "Abaixo do peso",
        "Normal_Weight": "Peso normal",
        "Overweight_Level_I": "Sobrepeso – Grau I",
        "Overweight_Level_II": "Sobrepeso – Grau II",
        "Obesity_Type_I": "Obesidade – Grau I",
        "Obesity_Type_II": "Obesidade – Grau II",
        "Obesity_Type_III": "Obesidade – Grau III"
    }

    st.subheader("📊 Resultado da Avaliação")
    st.success(f"Nível estimado de obesidade: **{label_map[prediction]}**")

    st.markdown("""
    ⚠️ **Aviso:**  
    Este sistema é um **apoio à decisão clínica** e não substitui avaliação médica.
    """)
