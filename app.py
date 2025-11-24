# hepatitis_api.py
import os
import pickle
import traceback
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

warnings.filterwarnings("ignore")

# Configuración del servidor
PORT = int(os.environ.get("PORT", 5000))
app = Flask("hepatitis_api")
CORS(app)

# Función para cargar modelos o normalizadores
def cargar_pickle(path):
    """Intenta cargar con joblib y si falla, con pickle."""
    try:
        from joblib import load as joblib_load
        return joblib_load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# Rutas de archivos
BASE_DIR = os.path.dirname(__file__)
MODELO_PATH = os.path.join(BASE_DIR, "modelo_regresion_logistica.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Cargar modelo y scaler
try:
    modelo = cargar_pickle(MODELO_PATH)
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    modelo = None

try:
    scaler = cargar_pickle(SCALER_PATH)
except Exception as e:
    print(f"[AVISO] No se pudo cargar el scaler: {e}")
    scaler = None

# Ruta principal
@app.route("/", methods=["GET"])
def inicio():
    return jsonify({
        "status": "online",
        "mensaje": "API de predicción de supervivencia lista",
        "instrucciones": "Enviar POST a /predict con JSON de features",
        "ejemplo": {"features": [1, 2, 3, 4], "o_dict": {"feature1": "..."}}
    })

# Preparar datos de entrada
def procesar_input(datos):
    """Convierte JSON en array de numpy compatible con el modelo."""
    if "features" in datos:
        return np.array(datos["features"], dtype=float).reshape(1, -1)

    # Intentar respetar el orden de features del modelo
    if hasattr(modelo, "feature_names_in_"):
        try:
            return np.array([float(datos[f]) for f in modelo.feature_names_in_]).reshape(1, -1)
        except Exception:
            pass

    # Por defecto, ordenar las claves
    valores = [float(datos[k]) for k in sorted(datos.keys())]
    return np.array(valores, dtype=float).reshape(1, -1)

# Endpoint de predicción
@app.route("/predict", methods=["POST"])
def predict():
    if modelo is None:
        return jsonify({"error": "Modelo no disponible"}), 500

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSON inválido o ausente"}), 400

    try:
        X = procesar_input(data)

        if scaler:
            try:
                X = scaler.transform(X)
            except Exception as e:
                print(f"[WARN] Error usando scaler: {e}")

        pred = int(modelo.predict(X)[0])
        etiquetas = {0: "Vive", 1: "Muere", 2: "Muere"}

        probabilidades = None
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X)[0]
            probabilidades = {etiquetas[i]: float(p) for i, p in enumerate(proba)}
            # Nuevo resultado basado en la mayor probabilidad
            resultado = max(probabilidades, key=probabilidades.get)
            valor_crudo = int(np.argmax(proba))
        else:
            resultado = etiquetas.get(pred, f"Clase {pred}")
            valor_crudo = pred

        return jsonify({
            "resultado": resultado,
            "valor_crudo": valor_crudo,
            "probabilidades": probabilidades
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Error durante la predicción",
            "detalle": str(e)
        }), 500


# Ejecutar servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
