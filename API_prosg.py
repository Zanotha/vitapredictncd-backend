# -*- coding: utf-8 -*-
"""
Flask API for NCD Prediction and User Registration (PostgreSQL Edition)
"""

import os
import pandas as pd
import uuid
import numpy as np
import psycopg2
import joblib
import hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import urlparse

# Initialize Flask App
app = Flask(__name__)
CORS(app, origins=["*"])

# PostgreSQL connection using DATABASE_URL from Heroku
database_url = os.getenv("DATABASE_URL", "postgresql://postgres:Support22!@localhost:5432/NCD_Predictive_Analysis")
url = urlparse(database_url)
conn = psycopg2.connect(
    database=url.path[1:],
    user=url.username,
    password=url.password,
    host=url.hostname,
    port=url.port
)

# Load Models
models = {
    "RandomForest": joblib.load("RandomForest_model.pkl"),
    "XGBoost": joblib.load("XGBoost_model.pkl"),
    "LogisticRegression": joblib.load("LogisticRegression_model.pkl"),
    "DecisionTree": joblib.load("DecisionTree_model.pkl"),
    "SVM": joblib.load("SVM_model.pkl"),
}

# Load Scaler and Encoders
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Diseases
DISEASES = [
    "HadHeartAttack", "HadStroke", "HadDiabetes", "HadKidneyDisease",
    "HadAsthma", "HadCOPD", "HadArthritis", "HadDepressiveDisorder",
    "HadSkinCancer", "Hypertension"
]

@app.route("/")
def home():
    return "Flask API is running!"

# -------------------------------
# Test Database Connection Endpoint
# -------------------------------
@app.route('/test_db')
def test_db():
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("SELECT * FROM users LIMIT 1;")
        row = cur.fetchone()
        conn.close()
        return f"DB connected! Row: {row}"
    except Exception as e:
        return f"DB error: {str(e)}"

# -------------------------------
# Login Endpoint
# -------------------------------
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"}), 400

        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        print(f"🔐 Login attempt for email: {email}")
        print(f"🔑 Hashed input password: {hashed_password}")

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, hashed_password))
        user = cursor.fetchone()
        cursor.close()

        if user:
            print("✅ Login successful")
            return jsonify({"success": True, "message": "Login successful"})
        else:
            print("❌ Invalid credentials")
            return jsonify({"success": False, "message": "Invalid credentials"}), 401

    except Exception as e:
        print("❌ Login error:", e)
        return jsonify({"success": False, "message": "Internal server error", "error": str(e)}), 500

# -------------------------------
# Register Endpoint
# -------------------------------
@app.route("/register", methods=["POST"])
def register_user():
    try:
        data = request.get_json()
        print("Received registration data:", data)

        first_name = data.get("first_name", "").strip()
        last_name = data.get("last_name", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()

        print("Parsed values:", first_name, last_name, email)

        if not all([first_name, last_name, email, password]):
            return jsonify({"success": False, "message": "All fields are required"}), 400

        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            return jsonify({"success": False, "message": "Email already exists"}), 409

        cursor.execute(
            "INSERT INTO users (first_name, last_name, email, password) VALUES (%s, %s, %s, %s)",
            (first_name, last_name, email, hashed_password)
        )
        conn.commit()
        cursor.close()

        return jsonify({"success": True, "message": "User registered successfully!"})

    except Exception as e:
        conn.rollback()
        print("❌ Registration error:", e)
        return jsonify({
            "success": False,
            "message": "Internal server error",
            "error": str(e)
        }), 500

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.get_json()
        if not user_input:
            return jsonify({"error": "No data provided"}), 400

        # Extract PatientID and AGE, remove them from user_input
        patient_id = str(user_input.pop("PatientID"))
        age = user_input.pop("Age")
        uid = str(uuid.uuid4())

        # Prepare DataFrame for prediction
        df = pd.DataFrame([user_input])

        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else encoder.transform([encoder.classes_[0]])[0])

        df_scaled = scaler.transform(df)

        # Make predictions
        predictions = {}
        for name, model in models.items():
            preds = model.predict(df_scaled)[0]
            predictions[name] = {DISEASES[i]: int(preds[i]) for i in range(len(DISEASES))}

        # Insert into medical_history
        cursor = conn.cursor()

        # Prepare columns and values
        columns = list(user_input.keys())
        values = list(user_input.values())

        # Quote all column names to match the exact case in the schema
        quoted_columns = [f'"{col}"' for col in columns]
        placeholders = ", ".join(["%s"] * len(values))

        # Construct query with all quoted columns
        sql_query = f'INSERT INTO medical_history (uuid, "PatientID", "AGE", {", ".join(quoted_columns)}) VALUES (%s, %s, %s, {placeholders})'
        cursor.execute(sql_query, [uid, patient_id, age] + values)

        # Insert predictions into prediction_results
        for model_name, preds in predictions.items():
            for disease, prediction in preds.items():
                cursor.execute(
                    'INSERT INTO prediction_results (uuid, model_name, condition_name, prediction) VALUES (%s, %s, %s, %s)',
                    (uid, model_name, disease, prediction)
                )

        conn.commit()
        cursor.close()

        return jsonify({"predictions": predictions, "uuid": uid})

    except Exception as e:
        conn.rollback()
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500

# -------------------------------
# Medical History Endpoint
# -------------------------------
@app.route("/medical_history", methods=["GET"])
def get_history():
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM medical_history")
        medical_rows = cursor.fetchall()
        medical_columns = [column[0] for column in cursor.description]
        medical_history = [dict(zip(medical_columns, row)) for row in medical_rows]

        cursor.execute("SELECT * FROM prediction_results")
        predict_rows = cursor.fetchall()
        predict_columns = [column[0] for column in cursor.description]
        prediction_results = [dict(zip(predict_columns, row)) for row in predict_rows]

        predictions_by_uuid = {}
        for pred in prediction_results:
            uid = pred.get("uuid")
            if uid not in predictions_by_uuid:
                predictions_by_uuid[uid] = []
            predictions_by_uuid[uid].append({
                "uuid": pred.get("uuid"),
                "model": pred.get("model_name"),
                "condition": pred.get("condition_name"),
                "probability": pred.get("prediction"),
            })

        for record in medical_history:
            uid = record.get("uuid")
            record["predictions"] = predictions_by_uuid.get(uid, [])

        cursor.close()
        return jsonify({"medical_history": medical_history})

    except Exception as e:
        print("Medical history error:", str(e))
        return jsonify({"error": str(e)}), 500
