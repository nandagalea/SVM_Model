from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
model = joblib.load("svm_rbf_best_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", [])
    if not features:
        return jsonify({"error": "No features provided"}), 400
    try:
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "SVM API is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
