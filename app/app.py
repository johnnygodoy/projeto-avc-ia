import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ===============================
# LOAD CSS
# ===============================
def load_css():
    with open("assets/style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Cardio Risk AI",
    layout="wide",
    page_icon="🫀"
)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ===============================
# HEADER
# ===============================
st.markdown("""
<h1 class='main-title'>
🫀 Sistema Inteligente de Predição de Risco Cardiovascular com Inteligência Artificial
</h1>

<p class='subtitle'>
Modelo de Machine Learning para apoio à decisão clínica
</p>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("📋 Dados do Paciente")

age = st.sidebar.slider("Idade", 18, 100, 45)

gender_label = st.sidebar.selectbox("Sexo", ["Masculino", "Feminino"])
gender = 1 if gender_label == "Masculino" else 2

height = st.sidebar.slider("Altura (cm)", 140, 210, 170)
weight = st.sidebar.slider("Peso (kg)", 40, 150, 70)

ap_hi = st.sidebar.slider("Pressão Sistólica", 80, 240, 120)
ap_lo = st.sidebar.slider("Pressão Diastólica", 50, 140, 80)

# ===============================
# CAMPOS CLÍNICOS
# ===============================
colesterol_label = st.sidebar.selectbox(
    "Colesterol",
    ["Normal", "Acima do normal", "Muito alto"]
)

colesterol = {"Normal":1,"Acima do normal":2,"Muito alto":3}[colesterol_label]

glicose_label = st.sidebar.selectbox(
    "Glicose",
    ["Normal", "Acima do normal", "Muito alto"]
)

gluc = {"Normal":1,"Acima do normal":2,"Muito alto":3}[glicose_label]

fumante_label = st.sidebar.selectbox("Fumante", ["Não", "Sim"])
smoke = 1 if fumante_label == "Sim" else 0

alcool_label = st.sidebar.selectbox(
    "Consumo de Álcool",
    ["Não bebe", "Social", "Frequente"]
)

alco = {"Não bebe":0,"Social":1,"Frequente":1}[alcool_label]

atividade_label = st.sidebar.selectbox("Atividade Física", ["Não", "Sim"])
active = 1 if atividade_label == "Sim" else 0

# ===============================
# FEATURE ENGINEERING
# ===============================
age_years = age
bmi = weight / ((height / 100) ** 2)

pulse_pressure = ap_hi - ap_lo
mean_pressure = (ap_hi + ap_lo) / 2

high_pressure = int(ap_hi > 140)
high_cholesterol = int(colesterol > 1)
high_glucose = int(gluc > 1)

risk_score = high_pressure + high_cholesterol + high_glucose + smoke
pressure_age = ap_hi * age_years

age_group = 0 if age <= 40 else 1 if age <= 55 else 2 if age <= 70 else 3

# ===============================
# INPUT DATA
# ===============================
input_data = pd.DataFrame([{
    "age": age * 365,
    "gender": gender,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": colesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "age_years": age_years,
    "bmi": bmi,
    "pulse_pressure": pulse_pressure,
    "high_pressure": high_pressure,
    "high_cholesterol": high_cholesterol,
    "high_glucose": high_glucose,
    "risk_score": risk_score,
    "mean_pressure": mean_pressure,
    "pressure_age": pressure_age,
    "age_group": age_group
}])

# ===============================
# BOTÃO
# ===============================
if st.sidebar.button("🔍 Analisar Risco"):

    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.3:
        risk_level = "BAIXO RISCO"
        color = "green"
    elif prob < 0.6:
        risk_level = "MÉDIO RISCO"
        color = "orange"
    else:
        risk_level = "ALTO RISCO"
        color = "red"

    # ===============================
    # LAYOUT PRINCIPAL
    # ===============================
    col_left, col_right = st.columns([1,1])

    with col_left:
        st.metric("Probabilidade de Risco", f"{prob:.2%}")
        st.markdown(f"## {risk_level}")

    with col_right:
        col1, col_center, col3 = st.columns([1,2,1])
        with col_center:
            fig, ax = plt.subplots(figsize=(5,2))
            ax.barh(["Risco"], [prob], color=color)
            ax.set_xlim(0,1)
            ax.axis('off')
            ax.set_title("Nível de Risco")
            st.pyplot(fig)

    # ===============================
    # FATORES
    # ===============================
    st.markdown("---")
    st.subheader("📊 Fatores de Risco")

    factors = {
        "Pressão": ap_hi,
        "Colesterol": colesterol,
        "Idade": age,
        "BMI": round(bmi,1),
        "Score": risk_score
    }

    df_factors = pd.DataFrame(list(factors.items()), columns=["Fator","Valor"])
    df_factors = df_factors.sort_values(by="Valor", ascending=False)

    fig, ax = plt.subplots(figsize=(6,2))
    ax.bar(df_factors["Fator"], df_factors["Valor"], color="#4DA6FF")

    ax.set_yticks([])
    ax.set_title("Fatores de Risco")

    for spine in ["top","right","left"]:
        ax.spines[spine].set_visible(False)

    plt.xticks(rotation=0)
    plt.tight_layout()

    col1, col_center, col3 = st.columns([1,2,1])
    with col_center:
        st.pyplot(fig)

    # ===============================
    # INTERPRETAÇÃO
    # ===============================
    st.markdown("---")
    st.subheader("🤖 Interpretação do Modelo")

    st.write(f"""
    • Pressão: {ap_hi} mmHg  
    • Colesterol: {colesterol_label}  
    • Glicose: {glicose_label}  
    • Idade: {age} anos  
    • BMI: {round(bmi,1)}  
    • Score: {risk_score}
    """)

    st.markdown("---")
    st.warning("Sistema de apoio. Não substitui diagnóstico médico.")