import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Dividir el conjunto de datos en características (X) y variables objetivo (y)
X = df.drop(columns=['Enfermedad_Bronquitis',
                     'AccionesPreventivas_Mejorar la dieta',
                     'Recomendaciones_Visitar al médico'])  # Ajusta según tus variables objetivo

# Variables objetivo
y_enfermedad = df['Enfermedad_Bronquitis']  # Variable objetivo de la enfermedad
y_acciones = df['AccionesPreventivas_Mejorar la dieta']  # Variable objetivo de acciones
y_recomendaciones = df['Recomendaciones_Visitar al médico']  # Variable objetivo de recomendaciones

# Aplicar SMOTE para cada variable objetivo
smote = SMOTE(random_state=42)

# Primero, aplicar SMOTE para la enfermedad
X_resampled_enfermedad, y_resampled_enfermedad = smote.fit_resample(X, y_enfermedad)

# Aplicar SMOTE para las acciones preventivas
X_resampled_acciones, y_resampled_acciones = smote.fit_resample(X, y_acciones)

# Aplicar SMOTE para las recomendaciones
X_resampled_recomendaciones, y_resampled_recomendaciones = smote.fit_resample(X, y_recomendaciones)

# Dividir en conjuntos de entrenamiento y prueba para enfermedad
X_train_enfermedad, X_test_enfermedad, y_train_enfermedad, y_test_enfermedad = train_test_split(X_resampled_enfermedad, y_resampled_enfermedad, test_size=0.2, random_state=42)

# Dividir en conjuntos de entrenamiento y prueba para acciones preventivas
X_train_acciones, X_test_acciones, y_train_acciones, y_test_acciones = train_test_split(X_resampled_acciones, y_resampled_acciones, test_size=0.2, random_state=42)

# Dividir en conjuntos de entrenamiento y prueba para recomendaciones
X_train_recomendaciones, X_test_recomendaciones, y_train_recomendaciones, y_test_recomendaciones = train_test_split(X_resampled_recomendaciones, y_resampled_recomendaciones, test_size=0.2, random_state=42)

# Entrenar modelos
modelo_enfermedad = RandomForestClassifier(random_state=42)
modelo_enfermedad.fit(X_train_enfermedad, y_train_enfermedad)

modelo_acciones = RandomForestClassifier(random_state=42)
modelo_acciones.fit(X_train_acciones, y_train_acciones)

modelo_recomendaciones = RandomForestClassifier(random_state=42)
modelo_recomendaciones.fit(X_train_recomendaciones, y_train_recomendaciones)

# PRUEBA: Datos de entrada para realizar la predicción
datos_de_entrada = pd.DataFrame({
    'HipertensiónPrevia': [1],
    'HistPreeclampsia': [0],
    'Diabetes': [1],
    'EnfermedadRenal': [1],
    'Edad': [34],
    'Peso': [63],
    'Talla': [166],
    'Etnia': [3],
    'PASistolicammHg': [2],
    'PADiastolicammHg': [2],
    'ProteinaOrina': [2],
    'GananciaPesoKg': [13],
    'EdadGestacional': [18],
    'NumFetos': [1],
    'NivelActividadFisica': [1],
    'Dieta': [1],
    'ConsumoTabaco': [0],
    'ConsumoAlcohol': [0]
})

# Asegurarse de que las columnas coincidan con el conjunto de entrenamiento
datos_de_entrada = datos_de_entrada.reindex(columns=X_train_enfermedad.columns, fill_value=0)

# Predicción de enfermedad
probabilidad_enfermedad = modelo_enfermedad.predict_proba(datos_de_entrada)[:, 1] * 100
print(f"Probabilidad de tener bronquitis: {probabilidad_enfermedad[0]:.2f}%")

# Predicción de acciones preventivas
acciones = modelo_acciones.predict(datos_de_entrada)
if acciones[0] == 1:  # Si el modelo predice que "Mejorar la dieta" es necesario
    print("Acciones preventivas: Mejorar la dieta")
else:
    print("Acciones preventivas: Otra acción")

# Predicción de recomendaciones
recomendaciones = modelo_recomendaciones.predict(datos_de_entrada)
if recomendaciones[0] == 1:  # Si el modelo predice "Visitar al médico"
    print("Recomendación: Visitar al médico")
else:
    print("Recomendación: Otra recomendación")

# Evaluación del modelo
y_pred_enfermedad = modelo_enfermedad.predict(X_test_enfermedad)
y_pred_acciones = modelo_acciones.predict(X_test_acciones)
y_pred_recomendaciones = modelo_recomendaciones.predict(X_test_recomendaciones)

# Reporte de clasificación
print("Reporte de clasificación para Enfermedad:")
print(classification_report(y_test_enfermedad, y_pred_enfermedad))

print("Reporte de clasificación para Acciones Preventivas:")
print(classification_report(y_test_acciones, y_pred_acciones))

print("Reporte de clasificación para Recomendaciones:")
print(classification_report(y_test_recomendaciones, y_pred_recomendaciones))
