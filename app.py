import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Sistema Preditivo de Obesidade")

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

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    )

    model.fit(X, y)
    return model

model = train_model()
