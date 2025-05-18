#  Prédiction de Défaut de Crédit avec Streamlit

## 🎯 Objectif
Développer une application web interactive pour prédire le risque de défaut à partir de données financières.

## 🗃 Données
- **Source** : LendingClub (https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Classes** : 
  - `0` : Fully Paid (prêt remboursé)
  - `1` : Charged Off (défaut de paiement)

## 🧠 Modèle
- **Algorithme** : RandomForestClassifier
- **Prétraitement** :
  - Nettoyage et sélection des variables
  - Encodage des variables catégorielles
  - Standardisation des données numériques
  - Rééquilibrage avec SMOTE
- **Performances** :
  - Précision : 88%
  - Rappel : 86%
  - F1-score : 87%
  - Équilibre parfait entre les classes grâce à SMOTE
 
    ## Importance des variables
-  int_rate (taux d’intérêt) : plus influente
-  annual_inc (revenu annuel) : forte contribution
- dti (dette/revenu) : très significatif
- loan_amnt et emp_length : impact modéré
- purpose(Objet du crédit) : effet faible
- term` : influence minimale


## 🚀 Lancer l’Application
```bash
streamlit run app.py
```

## 🗂 Structure du Dépôt
```
projet_credit_risque/
├── app.py
├── loan.csv
├── credit_model_rf_smote.pkl
├── scaler_rf_smote.pkl
├── label_encoders_rf_smote.pkl
├── README.md
└── capture_demo.mp4
```

## ✅ Installation des dépendances
Créer un environnement virtuel puis :
```bash
pip install -r requirements.txt
```
