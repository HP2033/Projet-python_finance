import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger modèle, scaler, et encodeurs
model = joblib.load("credit_model_rf_smote.pkl")
scaler = joblib.load("scaler_rf_smote.pkl")
label_encoders = joblib.load("label_encoders_rf_smote.pkl")

st.title("Prédiction de Défaut de Crédit")
st.markdown("Cette application prédit si un client risque de faire défaut ou non.")

# Formulaire de saisie
loan_amnt = st.number_input("Montant du prêt ($)", min_value=500, max_value=50000, value=10000)
term = st.selectbox("Durée", label_encoders['term'].classes_)
int_rate = st.slider("Taux d'intérêt (%)", min_value=0.0, value=10.0)
emp_length = st.selectbox("Ancienneté (emploi)", label_encoders['emp_length'].classes_)
annual_inc = st.number_input("Revenu annuel ($)", min_value=1000, max_value=1000000, value=50000)
purpose = st.selectbox("Objet du prêt", label_encoders['purpose'].classes_)
debt = st.number_input("Montant actuel de la dette ($)", min_value=0.0, value=10000.0)

# Évite la division par zéro
if annual_inc > 0:
    dti = (debt / (annual_inc / 12))  # dette mensuelle / revenu mensuel
    dti = round(dti, 2)
else:
    dti = 0
    st.warning("⚠️ Revenu annuel invalide pour calculer le DTI.")
    
st.write(f"📊 DTI calculé automatiquement : **{dti:.2f}**")


# Transformer les catégories
term_enc = label_encoders['term'].transform([term])[0]
emp_length_enc = label_encoders['emp_length'].transform([emp_length])[0]
purpose_enc = label_encoders['purpose'].transform([purpose])[0]

# Créer le tableau final
X_new = pd.DataFrame([[
    loan_amnt, term_enc, int_rate, emp_length_enc,
    annual_inc, purpose_enc, dti
]], columns=[
    'loan_amnt', 'term', 'int_rate', 'emp_length',
    'annual_inc', 'purpose', 'dti'
])

# Standardiser
X_new_scaled = scaler.transform(X_new)

# Prédire
if st.button("Prédire le défaut"):
    pred = model.predict(X_new_scaled)[0]
    proba = model.predict_proba(X_new_scaled)[0][1]
    
    if pred == 1:
        st.error(f"⚠️ Risque élevé de défaut de paiement ({proba:.2%})")
    else:
        st.success(f"✅ Faible risque de défaut ({proba:.2%})")

    # Affichage graphique : score de risque
    st.subheader("Visualisation du score de risque (%)")

    score = int(proba * 100)
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