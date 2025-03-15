# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Cargar los datos
df = pd.read_csv("DATOS.csv", low_memory=False)

# 2. Transformar la Variable Objetivo
df["loan_default"] = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else (0 if x == "Fully Paid" else None))

# Eliminar préstamos con estado "Current" (en curso)
df = df.dropna(subset=["loan_default"])

# 3. Seleccionar Variables Clave
num_vars = ["loan_amnt", "funded_amnt", "int_rate", "installment", "pub_rec_bankruptcies", "tax_liens"]
cat_vars = ["grade", "sub_grade"]

# Rellenar valores nulos
df["pub_rec_bankruptcies"].fillna(0, inplace=True)
df["tax_liens"].fillna(0, inplace=True)

# Convertir `int_rate` de porcentaje a número decimal
df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float) / 100

# 4. Convertir Variables Categóricas a Números
df["grade"] = LabelEncoder().fit_transform(df["grade"])
df["sub_grade"] = LabelEncoder().fit_transform(df["sub_grade"])

# 5. Escalar Variables Numéricas
scaler = StandardScaler()
df[num_vars] = scaler.fit_transform(df[num_vars])

# 6. Dividir en Conjuntos de Entrenamiento y Prueba
X = df[num_vars + ["grade", "sub_grade"]]
y = df["loan_default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Entrenar el Modelo de Regresión Logística
modelo = LogisticRegression(max_iter=500, random_state=42)
modelo.fit(X_train, y_train)

# 8. Hacer Predicciones
y_pred = modelo.predict(X_test)
y_pred_proba = modelo.predict_proba(X_test)[:, 1]  # Probabilidad de incumplimiento

# 9. Evaluar el Modelo
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("\nAUC-ROC Score:")
print(roc_auc_score(y_test, y_pred_proba))








