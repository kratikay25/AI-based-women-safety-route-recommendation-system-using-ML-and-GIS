from pathlib import Path
import math

import networkx as nx
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# 1. FILE PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "merged_dataset.csv"
GRAPH_PATH = BASE_DIR / "delhi_road_network.graphml"

df = None
G = None

FEATURES = [
    "Crime_Index",
    "Crime_Density",
    "Total_Crimes",
    "Police_Distance_m",
    "Hospital_Distance_m"
]

FEATURE_WEIGHTS = {
    "Crime_Index": 0.35,
    "Crime_Density": 0.25,
    "Total_Crimes": 0.15,
    "Police_Distance_m": 0.15,
    "Hospital_Distance_m": 0.10
}

feature_lookup = {}

# --------------------------------------------------
# 2. LOAD DATASET AND ROAD NETWORK
# --------------------------------------------------

try:
    df = pd.read_csv(DATASET_PATH, dtype={"u": str, "v": str})

    required_columns = ["u", "v", "length"] + FEATURES
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    for col in ["length"] + FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns)

    if df.empty:
        raise ValueError("The dataset contains no usable road records.")

    # Aggregate duplicate directed road pairs using median values.
    grouped = (
        df.groupby(["u", "v"], as_index=False)[FEATURES]
        .median()
    )

    # Calculate robust normalization limits from the dataset.
    normalization = {}

    for feature in FEATURES:
        low = float(df[feature].quantile(0.05))
        high = float(df[feature].quantile(0.95))

        if high <= low:
            high = low + 1.0

        normalization[feature] = (low, high)

    # Store ORIGINAL feature values for each directed road pair.
    # Normalization happens later, when graph edges are processed.
    for row in grouped.itertuples(index=False):
        values = {
            feature: float(getattr(row, feature))
            for feature in FEATURES
        }

        feature_lookup[(str(row.u), str(row.v))] = values

    G = nx.read_graphml(GRAPH_PATH)

    if G.number_of_nodes() == 0:
        raise ValueError("The road network contains no nodes.")

    # Attach safety features and calculate composite risk for each edge.
    default_values = {
        feature: float(df[feature].median())
        for feature in FEATURES
    }

    def normalize_features(raw_values):
        normalized = {}

        for feature in FEATURES:
            low, high = normalization[feature]
            clipped = min(max(raw_values[feature], low), high)
            normalized[feature] = (clipped - low) / (high - low)

        return normalized

    for u, v, key, attrs in G.edges(keys=True, data=True):
        try:
            length = float(attrs.get("length", 1.0))
        except (TypeError, ValueError):
            length = 1.0

        if not math.isfinite(length) or length <= 0:
            length = 1.0

        attrs["length"] = length

        raw_values = feature_lookup.get(
            (str(u), str(v)),
            feature_lookup.get((str(v), str(u)), default_values)
        )

        normalized = normalize_features(raw_values)

        for feature in FEATURES:
            attrs[feature.lower()] = float(raw_values[feature])

        risk = sum(
            normalized[feature] * FEATURE_WEIGHTS[feature]
            for feature in FEATURES
        )

        attrs["risk_score"] = float(risk)

        # Distance cost increases with estimated composite risk.
        attrs["safe_cost"] = length * (1 + 2 * risk)

    print(f"Road network loaded: {G.number_of_nodes()} nodes")
    print(f"Road network edges: {G.number_of_edges()}")

except Exception as exc:
    print(f"Startup error: {exc}")
    df = None
    G = None


# --------------------------------------------------
# 3. FIND THE NEAREST ROAD-NETWORK NODE
# --------------------------------------------------

def nearest_node(latitude, longitude):
    """
    Find the closest graph node to the supplied coordinates.

    Uses an approximate coordinate-distance calculation.
    """

    best_node = None
    best_distance = float("inf")

    for node, attrs in G.nodes(data=True):
        try:
            lat = float(attrs["y"])
            lon = float(attrs["x"])
        except (KeyError, TypeError, ValueError):
            continue

        distance = (
            (lat - latitude) ** 2
            + (lon - longitude) ** 2
        )

        if distance < best_distance:
            best_distance = distance
            best_node = node

    if best_node is None:
        raise ValueError(
            "No usable road-network nodes were found."
        )

    return best_node


# --------------------------------------------------
# 4. CALCULATE ROUTE DETAILS
# --------------------------------------------------

def route_details(path, optimization="length"):
    """Calculate route details using the composite safety score."""

    total_distance = 0.0
    weighted_risk = 0.0
    weighted_crime = 0.0
    weighted_features = {feature: 0.0 for feature in FEATURES}
    coordinates = []

    for node in path:
        attrs = G.nodes[node]
        coordinates.append({
            "latitude": float(attrs["y"]),
            "longitude": float(attrs["x"])
        })

    for u, v in zip(path, path[1:]):
        edges = G.get_edge_data(u, v)

        if not edges:
            continue

        def edge_cost(attrs):
            try:
                value = float(attrs.get(optimization, 1.0))
                return value if math.isfinite(value) else float("inf")
            except (TypeError, ValueError):
                return float("inf")

        edge = min(edges.values(), key=edge_cost)

        length = float(edge["length"])
        risk = float(edge["risk_score"])
        crime = float(edge["crime_index"])

        total_distance += length
        weighted_risk += length * risk
        weighted_crime += length * crime

        for feature in FEATURES:
            weighted_features[feature] += (
                length * float(edge[feature.lower()])
            )

    if total_distance > 0:
        average_risk = weighted_risk / total_distance
        average_crime = weighted_crime / total_distance

        average_features = {
            feature: round(value / total_distance, 3)
            for feature, value in weighted_features.items()
        }
    else:
        average_risk = 1.0
        average_crime = float(df["Crime_Index"].median())
        average_features = {}

    safety_score = round((1 - average_risk) * 100)
    safety_score = max(0, min(100, safety_score))

    if safety_score >= 67:
        crime_risk = "Low"
    elif safety_score >= 34:
        crime_risk = "Medium"
    else:
        crime_risk = "High"

    estimated_minutes = (
        max(1, math.ceil(total_distance / 5000 * 60))
        if total_distance > 0 else 0
    )

    return {
        "distance_m": round(total_distance, 2),
        "distance_km": round(total_distance / 1000, 2),
        "estimated_time_minutes": estimated_minutes,
        "average_crime_index": round(average_crime, 2),
        "average_risk_score": round(average_risk, 3),
        "average_safety_features": average_features,
        "safety_score": safety_score,
        "crime_risk": crime_risk,
        "coordinates": coordinates
    }


# 5. HOME ENDPOINT
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Women's Safety Route Backend Running",
        "dataset_loaded": df is not None,
        "graph_loaded": G is not None
    })


# --------------------------------------------------
# 6. DATASET INFORMATION ENDPOINT
# --------------------------------------------------

@app.route("/dataset-info", methods=["GET"])
def dataset_info():

    if df is None or G is None:
        return jsonify({
            "error": "Dataset or road network failed to load."
        }), 500

    return jsonify({
        "rows": len(df),
        "columns": list(df.columns),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges()
    })


# --------------------------------------------------
# 7. SAFEST ROUTE ENDPOINT
# --------------------------------------------------

@app.route("/safest-route", methods=["POST"])
def safest_route():

    if df is None or G is None:
        return jsonify({
            "error": "Dataset or road network is unavailable."
        }), 500

    data = request.get_json(silent=True) or {}

    # Read and validate the coordinates
    try:
        source_lat = float(data["source_lat"])
        source_lon = float(data["source_lon"])
        destination_lat = float(data["destination_lat"])
        destination_lon = float(data["destination_lon"])

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": (
                "Provide source_lat, source_lon, "
                "destination_lat and destination_lon as numbers."
            )
        }), 400

    # Reject invalid or non-finite coordinates
    coordinates_to_check = [
        source_lat,
        source_lon,
        destination_lat,
        destination_lon
    ]

    if not all(
        math.isfinite(value)
        for value in coordinates_to_check
    ):
        return jsonify({
            "error": "Coordinates must be finite numbers."
        }), 400

    # Approximate supported Delhi-area boundary
    for lat, lon in [
        (source_lat, source_lon),
        (destination_lat, destination_lon)
    ]:
        if not (
            28.0 <= lat <= 29.0
            and 76.5 <= lon <= 77.8
        ):
            return jsonify({
                "error": (
                    "Coordinates appear to be outside "
                    "the supported Delhi area."
                )
            }), 400

    try:
        # Snap the requested locations to nearby road nodes
        source_node = nearest_node(
            source_lat,
            source_lon
        )

        destination_node = nearest_node(
            destination_lat,
            destination_lon
        )

        if source_node == destination_node:
            return jsonify({
                "error": (
                    "Start and destination snap to the same "
                    "road junction. Choose points farther apart."
                )
            }), 400

        # Find the shortest route by road distance
        shortest_path = nx.shortest_path(
            G,
            source=source_node,
            target=destination_node,
            weight="length"
        )

        # Find the route minimizing distance plus crime penalty
        safer_path = nx.shortest_path(
            G,
            source=source_node,
            target=destination_node,
            weight="safe_cost"
        )

        # Return both route options and their details
        return jsonify({
            "source_node": source_node,
            "destination_node": destination_node,

            "shortest_route": route_details(
                shortest_path,
                optimization="length"
            ),

            "safer_route": route_details(
                safer_path,
                optimization="safe_cost"
            )
        })

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return jsonify({
            "error": (
                "No connected route was found "
                "between these locations."
            )
        }), 404

    except Exception as exc:
        app.logger.exception("Route calculation failed")

        return jsonify({
            "error": "Route calculation failed.",
            "details": str(exc)
        }), 500


# --------------------------------------------------
# 8. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)