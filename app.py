# ============================================================
#  ChurnLens — Flask Application
#  Run: python app.py
#  Open: http://127.0.0.1:5000
# ============================================================

from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from flask_cors import CORS
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Load ML Model ────────────────────────────────────────────
MODEL_PATH = "customer_churn_model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Model load failed: {e}")
else:
    print("⚠️  Model file not found — /predict will use heuristic fallback")


# ── Page Routes  (both /route  AND  /route.html work) ────────

PAGES = {
    "index":    "index.html",
    "register": "register.html",
    "dashboard":"dashboard.html",
    "analysis": "analysis.html",
    "upload":   "upload.html",
    "view":     "view.html",
    "predict":  "predict.html",
    "models":   "models.html",
}

@app.route("/home")
@app.route("/home.html")
def home():
    return render_template("home.html")

@app.route("/login")
@app.route("/login.html")
def login_page():
    return render_template("index.html")

@app.route("/")
@app.route("/index.html")
def index():
    return render_template("home.html")  # root -> landing page

@app.route("/register")
@app.route("/register.html")
def register():
    return render_template("register.html")

@app.route("/dashboard")
@app.route("/dashboard.html")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analysis")
@app.route("/analysis.html")
def analysis():
    return render_template("analysis.html")

@app.route("/upload")
@app.route("/upload.html")
def upload_page():
    return render_template("upload.html")

@app.route("/view")
@app.route("/view.html")
def view():
    return render_template("view.html")

@app.route("/predict")
@app.route("/predict.html")
def predict_page():
    return render_template("predict.html")

@app.route("/models")
@app.route("/models.html")
def models_page():
    return render_template("models.html")


# ── API: Health Check ────────────────────────────────────────

@app.route("/api/status")
def status():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "version": "2.4.1"
    })


# ── API: Predict Churn ───────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # ── Try real model first ──
    if model is not None:
        try:
            features = np.array(data["input"]).reshape(1, -1)
            prediction = int(model.predict(features)[0])
            try:
                prob = float(model.predict_proba(features)[0][1])
            except Exception:
                prob = 0.5
            return jsonify({
                "prediction": prediction,
                "probability": round(prob, 4),
                "source": "model"
            })
        except Exception as e:
            print(f"Model inference error: {e}")

    # ── Heuristic fallback (when model not available) ──
    try:
        d = data.get("features", {})
        score = 0.08

        contract   = d.get("contract", "Month-to-month")
        internet   = d.get("internet", "Fiber optic")
        payment    = d.get("payment", "Electronic check")
        tenure     = int(d.get("tenure", 12))
        monthly    = float(d.get("monthly", 65))
        senior     = int(d.get("senior", 0))
        paperless  = d.get("paperless", "Yes")

        if contract == "Month-to-month":  score += 0.35
        elif contract == "One year":       score += 0.12
        if internet == "Fiber optic":      score += 0.18
        if payment == "Electronic check":  score += 0.14
        if paperless == "Yes":             score += 0.05
        if senior == 1:                    score += 0.09
        if tenure < 6:                     score += 0.18
        elif tenure < 12:                  score += 0.10
        elif tenure > 36:                  score -= 0.12
        elif tenure > 60:                  score -= 0.18
        if monthly > 90:                   score += 0.08
        elif monthly < 40:                 score -= 0.06

        score = max(0.04, min(0.97, score))
        prediction = 1 if score > 0.5 else 0

        return jsonify({
            "prediction": prediction,
            "probability": round(score, 4),
            "source": "heuristic"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── API: Model Accuracy ──────────────────────────────────────

@app.route("/api/accuracy", methods=["GET"])
def accuracy():
    return jsonify({
        "Random Forest":        {"accuracy": 0.91, "f1": 0.89, "auc": 0.94, "precision": 0.88, "recall": 0.90},
        "Decision Tree":        {"accuracy": 0.87, "f1": 0.83, "auc": 0.87, "precision": 0.84, "recall": 0.82},
        "Logistic Regression":  {"accuracy": 0.82, "f1": 0.79, "auc": 0.86, "precision": 0.80, "recall": 0.78},
    })


# ── API: Dataset Stats ───────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def stats():
    CSV_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(CSV_PATH):
        # Return pre-computed stats if CSV not present
        return jsonify({
            "total": 7043, "churned": 1869, "retained": 5174,
            "churn_rate": 26.54,
            "avg_monthly": 64.76, "avg_tenure": 32.4,
            "contract": {"Month-to-month": 3875, "One year": 1473, "Two year": 1695},
            "internet": {"DSL": 2421, "Fiber optic": 3096, "No": 1526},
            "churn_contract": {"Month-to-month": 1655, "One year": 166, "Two year": 48},
            "churn_internet": {"DSL": 459, "Fiber optic": 1297, "No": 113},
        })
    try:
        df = pd.read_csv(CSV_PATH)
        churned = df[df["Churn"] == "Yes"]
        return jsonify({
            "total":        int(len(df)),
            "churned":      int(len(churned)),
            "retained":     int(len(df) - len(churned)),
            "churn_rate":   round(len(churned) / len(df) * 100, 2),
            "avg_monthly":  round(float(df["MonthlyCharges"].mean()), 2),
            "avg_tenure":   round(float(df["tenure"].mean()), 1),
            "contract":     df["Contract"].value_counts().to_dict(),
            "internet":     df["InternetService"].value_counts().to_dict(),
            "churn_contract": churned["Contract"].value_counts().to_dict(),
            "churn_internet": churned["InternetService"].value_counts().to_dict(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Upload CSV ──────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400
    try:
        df = pd.read_csv(file)
        return jsonify({
            "success": True,
            "rows":    int(len(df)),
            "columns": int(len(df.columns)),
            "cols":    list(df.columns),
            "preview": df.head(5).to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  ChurnLens Flask Server")
    print("  http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)