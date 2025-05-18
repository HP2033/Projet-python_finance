import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger modèle, scaler, encodeurs et ordre des colonnes
model = joblib.load("models/credit_model_rf_smote.pkl")
scaler = joblib.load("models/scaler_rf_smote.pkl")
label_encoders = joblib.load("models/label_encoders_rf_smote.pkl")
feature_order = joblib.load("models/feature_order.pkl")

st.title("🎯 Prédiction de Défaut de Crédit")
st.markdown("Cette application prédit si un client risque de faire défaut ou non.")

# Formulaire de saisie
loan_amnt = st.number_input("Montant du prêt ($)", min_value=500.0, max_value=35000.0, value=10000.0, step=500.0)
int_rate = st.slider("Taux d'intérêt (%)", min_value=5.32, max_value=28.99, value=10.0, step=0.1)
emp_length = st.selectbox("Ancienneté (emploi)", label_encoders['emp_length'].classes_)
annual_inc = st.number_input("Revenu annuel ($)", min_value=3000.0, max_value=8706582.0, value=50000.0, step=1000.0)
debt = st.number_input("Montant de la dette mensuelle ($)", min_value=0.0, max_value=annual_inc * 5)
purpose = st.selectbox("Objet du prêt", label_encoders['purpose'].classes_)
inq_last_6mths = st.number_input("Nombre de demandes de crédit (6 derniers mois)", min_value=0)

# Calcul automatique du DTI
if annual_inc > 0:
    dti = min(debt / (annual_inc / 12), 57.14)  # DTI limité à 57.14 comme dans le dataset
else:
    dti = 0.0
st.markdown(f"📊 DTI calculé automatiquement : **{dti:.2f} %**")

# Encodage
emp_length_enc = label_encoders['emp_length'].transform([emp_length])[0]
purpose_enc = label_encoders['purpose'].transform([purpose])[0]

# Création du DataFrame utilisateur
X_input = pd.DataFrame([[
    loan_amnt, dti, int_rate, emp_length_enc,
    annual_inc, purpose_enc, inq_last_6mths
]], columns=[
    'loan_amnt', 'dti', 'int_rate', 'emp_length',
    'annual_inc', 'purpose', 'inq_last_6mths'
])

# Réordonner les colonnes pour correspondre au modèle
X_input = X_input[feature_order]

# Standardiser
X_scaled = scaler.transform(X_input)

# Prédiction
if st.button("Prédire le défaut"):
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]

    score = int(proba * 100)
    if pred == 1:
        st.error(f"⚠️ Risque élevé de défaut de paiement ({score:.2f} %)")
    else:
        st.success(f"✅ Faible risque de défaut ({score:.2f} %)")

    # Barre de score de risque
    st.subheader("Visualisation du score de risque (%)")
    color = "green"
    if score > 70:
        color = "red"
    elif score > 40:
        color = "orange"

    st.markdown(f"""
    <div style="border: 1px solid #ccc; border-radius: 8px; padding: 10px;">
        <div style="font-weight: bold;">Score de risque : {score} %</div>
        <div style="background-color: #eee; border-radius: 5px; height: 25px; width: 100%;">
            <div style="background-color: {color}; width: {score}%; height: 100%; border-radius: 5px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
