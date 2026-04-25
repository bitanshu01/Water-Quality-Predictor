from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -----------------------------
# LOAD MODELS SAFELY
# -----------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(base_dir, "models")

print("Loading models from:", models_path)

xgb_model = joblib.load(os.path.join(models_path, "xgb_model.pkl"))
encoder = joblib.load(os.path.join(models_path, "station_encoder.pkl"))
scaler_X = joblib.load(os.path.join(models_path, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(models_path, "scaler_y.pkl"))
lstm_model = load_model(
    os.path.join(models_path, "lstm_model.keras"),
    compile=False
)

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("processed_data.csv")
df.columns = df.columns.str.strip().str.lower()

# Ensure lowercase consistency
df.columns = df.columns.str.strip().str.lower()

# Get unique areas
areas = sorted(df['locations_full'].unique())

# -----------------------------
# CLASSIFICATION
# -----------------------------

WQI_GOOD_MAX = 55
WQI_AVERAGE_MAX = 105

def classify_wqi(wqi):
    if wqi <= WQI_GOOD_MAX:
        return "Good"
    elif wqi <= WQI_AVERAGE_MAX:
        return "Average"
    else:
        return "Bad"

# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get-areas", methods=["GET"])
def get_areas():
    return jsonify(areas)

@app.route("/get-stations", methods=["POST"])
def get_stations():
    data = request.json
    area = data.get("area")

    stations = df[df['locations_full'] == area]['station code'].unique().tolist()

    return jsonify(stations)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        station = data.get("station")
        if not station:
            return jsonify({"error": "Station is required"}), 400

        station = str(station).strip()
        target_year = int(data.get("year"))

        if station not in df['station code'].unique():
            return jsonify({"error": "Invalid station code"}), 400

        station_encoded = encoder.transform([station])[0]

        station_data = df[df["station code"] == station].sort_values("year")

        historical_years = station_data["year"].tolist()
        historical_wqi = station_data["wqi"].tolist()

        first_wqi = station_data["wqi"].iloc[0]
        latest_wqi = station_data["wqi"].iloc[-1]

        last_year = station_data["year"].max()

        if target_year <= last_year:
            return jsonify({"error": "Please select a future year"}), 400

        if len(station_data) < 3:
            return jsonify({"error": "Not enough historical data"}), 400

        last_values = station_data["wqi"].values[-3:]

        # 🔥 CHEMICAL FEATURES
        last_ph = station_data["ph"].iloc[-1]
        last_do = station_data["do"].iloc[-1]
        last_bod = station_data["bod"].iloc[-1]
        last_nitrate = station_data["nitrate"].iloc[-1]

        years_to_predict = target_year - last_year
        final_pred = None

        for i in range(years_to_predict):

            next_year = last_year + i + 1

            # 🔥 ROLLING FEATURE
            roll3 = np.mean(last_values)

            # 🔥 XGBOOST
            xgb_input = np.array([[ 
                next_year,
                station_encoded,
                last_ph,
                last_do,
                last_bod,
                last_nitrate,
                last_values[-1],
                last_values[-2],
                last_values[-3],
                roll3
            ]])

            xgb_pred = xgb_model.predict(xgb_input)[0]

            # 🔥 FIXED LSTM INPUT (MATCH TRAINING)
            seq = np.array([
                [last_ph, last_do, last_bod, last_nitrate, last_values[-1]],
                [last_ph, last_do, last_bod, last_nitrate, last_values[-2]],
                [last_ph, last_do, last_bod, last_nitrate, last_values[-3]]
            ])

            seq_scaled = scaler_X.transform(seq.reshape(-1,5)).reshape(1,3,5)

            lstm_pred = scaler_y.inverse_transform(
                lstm_model.predict(seq_scaled, verbose=0)
            )[0][0]

            # 🔥 SMART HYBRID (IMPROVED)
            if abs(xgb_pred - lstm_pred) < 8:
                final_pred = 0.6 * xgb_pred + 0.4 * lstm_pred
            else:
                final_pred = xgb_pred

            last_values = np.append(last_values, final_pred)

        # -----------------------------
        # ANALYTICS
        # -----------------------------

        improvement_percent = ((first_wqi - latest_wqi) / first_wqi) * 100

        trend = "Improving" if latest_wqi < first_wqi else "Worsening"

        if final_pred <= 50:
            risk = "Low"
        elif final_pred <= 100:
            risk = "Moderate"
        else:
            risk = "High"

        return jsonify({
            "historical_years": historical_years,
            "historical_wqi": historical_wqi,
            "future_year": target_year,
            "predicted_wqi": round(float(final_pred), 2),
            "category": classify_wqi(final_pred),
            "improvement_percent": round(float(improvement_percent), 2),
            "trend": trend,
            "risk": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/station-ranking", methods=["GET"])
def station_ranking():

    latest_year = df["year"].max()

    latest_data = df[df["year"] == latest_year]

    ranking = latest_data.sort_values("wqi", ascending=False)

    top_5 = ranking.head(5)[["station code", "locations", "wqi"]]
    bottom_5 = ranking.tail(5)[["station code", "locations", "wqi"]]

    return jsonify({
        "year": int(latest_year),
        "most_polluted": top_5.to_dict(orient="records"),
        "cleanest": bottom_5.to_dict(orient="records")
    })
# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
