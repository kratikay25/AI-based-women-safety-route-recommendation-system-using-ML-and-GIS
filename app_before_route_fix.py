from pathlib import Path

import networkx as nx
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "merged_dataset.csv"
GRAPH_PATH = BASE_DIR / "delhi_road_network.graphml"

df = None
G = None
crime_lookup = {}

try:
    df = pd.read_csv(DATASET_PATH, dtype={"u": str, "v": str})
    df["Crime_Index"] = pd.to_numeric(df["Crime_Index"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce")

    df = df.dropna(subset=["u", "v", "Crime_Index", "length"])
    print(f"Dataset loaded: {len(df)} road records")

    for row in df.itertuples(index=False):
        crime_lookup[(str(row.u), str(row.v))] = float(row.Crime_Index)

    G = nx.read_graphml(GRAPH_PATH)

    # GraphML stores edge attributes as strings.
    for u, v, key, attrs in G.edges(keys=True, data=True):
        try:
            length = float(attrs.get("length", 1.0))
        except (TypeError, ValueError):
            length = 1.0

        if length <= 0:
            length = 1.0

        attrs["length"] = length

        crime = crime_lookup.get(
            (str(u), str(v)),
            crime_lookup.get((str(v), str(u)))
        )

        # Missing crime data gets the dataset average, not zero risk.
        if crime is None:
            crime = float(df["Crime_Index"].mean())

        attrs["crime_index"] = crime

        # Distance plus a crime-related penalty.
        attrs["safe_cost"] = length * (
            1 + crime / float(df["Crime_Index"].max())
        )

    print(f"Road network loaded: {G.number_of_nodes()} nodes")
    print(f"Road network edges: {G.number_of_edges()}")

except Exception as exc:
    print(f"Startup error: {exc}")
    df = None
    G = None


def nearest_node(latitude, longitude):
    """Return the closest graph node to the requested coordinates."""
    best_node = None
    best_distance = float("inf")

    for node, attrs in G.nodes(data=True):
        try:
            lat = float(attrs["y"])
            lon = float(attrs["x"])
        except (KeyError, TypeError, ValueError):
            continue

        # Approximate local distance for choosing the closest node.
        distance = (lat - latitude) ** 2 + (lon - longitude) ** 2

        if distance < best_distance:
            best_distance = distance
            best_node = node

    if best_node is None:
        raise ValueError("No usable road-network nodes were found.")

    return best_node


def route_details(path):
    """Calculate distance, crime index, score, time and map coordinates."""
    total_distance = 0.0
    crime_weighted_sum = 0.0
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

        # For parallel edges, use the edge with the lowest selected cost.
        edge = min(
            edges.values(),
            key=lambda attrs: float(attrs.get("length", 1.0))
        )

        length = float(edge["length"])
        crime = float(edge["crime_index"])

        total_distance += length
        crime_weighted_sum += length * crime

    if total_distance > 0:
        average_crime = crime_weighted_sum / total_distance
    else:
        average_crime = float(df["Crime_Index"].mean())

    max_crime = float(df["Crime_Index"].max())
    safety_score = round(
        max(0, min(100, (1 - average_crime / max_crime) * 100))
    )

    if average_crime < float(df["Crime_Index"].quantile(0.33)):
        crime_risk = "Low"
    elif average_crime < float(df["Crime_Index"].quantile(0.67)):
        crime_risk = "Medium"
    else:
        crime_risk = "High"

    # Rough walking-time estimate: 5 km/h.
    estimated_minutes = round(total_distance / 5000 * 60)

    return {
        "distance_m": round(total_distance, 2),
        "distance_km": round(total_distance / 1000, 2),
        "estimated_time_minutes": estimated_minutes,
        "average_crime_index": round(average_crime, 2),
        "safety_score": safety_score,
        "crime_risk": crime_risk,
        "coordinates": coordinates
    }


@app.route("/")
def home():
    return jsonify({
        "message": "Women’s Safety Route Backend Running",
        "dataset_loaded": df is not None,
        "graph_loaded": G is not None
    })


@app.route("/dataset-info")
def dataset_info():
    if df is None or G is None:
        return jsonify({"error": "Dataset or road network failed to load"}), 500

    return jsonify({
        "rows": len(df),
        "columns": list(df.columns),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges()
    })


@app.route("/safest-route", methods=["POST"])
def safest_route():
    if df is None or G is None:
        return jsonify({"error": "Dataset or road network is unavailable"}), 500

    data = request.get_json(silent=True) or {}

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

    # This graph covers Delhi; reject coordinates outside its broad area.
    for lat, lon in [
        (source_lat, source_lon),
        (destination_lat, destination_lon)
    ]:
        if not (28.0 <= lat <= 29.0 and 76.5 <= lon <= 77.8):
            return jsonify({
                "error": "Coordinates appear to be outside the supported Delhi area."
            }), 400

    try:
        source_node = nearest_node(source_lat, source_lon)
        destination_node = nearest_node(destination_lat, destination_lon)

        if source_node == destination_node:
            return jsonify({
                "error": (
                    "Start and destination snap to the same road junction. "
                    "Choose points farther apart."
                )
            }), 400

        shortest_path = nx.shortest_path(
            G, source_node, destination_node, weight="length"
        )

        safer_path = nx.shortest_path(
            G, source_node, destination_node, weight="safe_cost"
        )

        return jsonify({
            "source_node": source_node,
            "destination_node": destination_node,
            "shortest_route": route_details(shortest_path),
            "safer_route": route_details(safer_path)
        })

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return jsonify({
            "error": "No connected route was found between these locations."
        }), 404
    except Exception as exc:
        app.logger.exception("Route calculation failed")
        return jsonify({"error": "Route calculation failed", "details": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)