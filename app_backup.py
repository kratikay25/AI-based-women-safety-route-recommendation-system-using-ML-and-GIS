from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load Dataset
try:
    df = pd.read_csv(
        "/Users/kratikay/Desktop/College/MINOR PROJECT/data/merged_dataset.csv"
    )
    print("Dataset Loaded Successfully")
    print("Rows:", len(df))
except Exception as e:
    print("Dataset Error:", e)
    df = None


@app.route("/")
def home():
    return "Backend Running Successfully"


@app.route("/dataset-info")
def dataset_info():

    if df is None:
        return jsonify({
            "error": "Dataset not loaded"
        })

    return jsonify({
        "rows": len(df),
        "columns": list(df.columns)
    })


@app.route("/safest-route", methods=["POST"])
def safest_route():

    data = request.json

    source = data.get("source")
    destination = data.get("destination")

    if df is not None:

        max_crime = df["Crime_Index"].max()
        avg_crime = df["Crime_Index"].mean()

        safety_score = round(
        (1 - (avg_crime / max_crime)) * 100
     )

        if safety_score >= 80:
            crime_risk = "Low"
        elif safety_score >= 50:
            crime_risk = "Medium"
        else:
            crime_risk = "High"

    else:
        avg_crime = 0
        safety_score = 92
        crime_risk = "Low"

    print("Average Crime Index:", avg_crime)
    print("Safety Score:", safety_score)

    return jsonify({
        "safety_score": safety_score,
        "crime_risk": crime_risk,
        "distance": "4.8 km",
        "estimated_time": "12 mins",
        "source": source,
        "destination": destination
    })


if __name__ == "__main__":
    app.run(debug=True)