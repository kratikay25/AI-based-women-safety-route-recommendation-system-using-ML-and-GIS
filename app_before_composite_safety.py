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
crime_lookup = {}

# --------------------------------------------------
# 2. LOAD DATASET AND ROAD NETWORK
# --------------------------------------------------

try:
    # Load the road dataset
    df = pd.read_csv(
        DATASET_PATH,
        dtype={"u": str, "v": str}
    )

    # Convert required columns to numeric values
    df["Crime_Index"] = pd.to_numeric(
        df["Crime_Index"], errors="coerce"
    )

    df["length"] = pd.to_numeric(
        df["length"], errors="coerce"
    )

    # Remove records without required values
    df = df.dropna(
        subset=["u", "v", "Crime_Index", "length"]
    )

    if df.empty:
        raise ValueError("The dataset contains no usable road records.")

    if float(df["Crime_Index"].max()) <= 0:
        raise ValueError("Crime_Index must contain a positive maximum value.")

    print(f"Dataset loaded: {len(df)} road records")

    # Map each directed road segment to its crime index
    for row in df.itertuples(index=False):
        crime_lookup[(str(row.u), str(row.v))] = float(
            row.Crime_Index
        )

    # Load the road network
    G = nx.read_graphml(GRAPH_PATH)

    if G.number_of_nodes() == 0:
        raise ValueError("The road network contains no nodes.")

    # Convert road attributes and calculate route costs
    average_crime = float(df["Crime_Index"].mean())
    maximum_crime = float(df["Crime_Index"].max())

    for u, v, key, attrs in G.edges(keys=True, data=True):

        # Road length
        try:
            length = float(attrs.get("length", 1.0))
        except (TypeError, ValueError):
            length = 1.0

        if not math.isfinite(length) or length <= 0:
            length = 1.0

        attrs["length"] = length

        # Crime index for this road segment
        crime = crime_lookup.get(
            (str(u), str(v)),
            crime_lookup.get(
                (str(v), str(u)),
                average_crime
            )
        )

        attrs["crime_index"] = float(crime)

        # Distance-based cost with a crime penalty
        attrs["safe_cost"] = length * (
            1 + float(crime) / maximum_crime
        )

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
    """
    Calculate route distance, crime index, safety score,
    estimated walking time, risk category and map coordinates.

    optimization:
        length    -> shortest-distance route
        safe_cost -> safer-cost route
    """

    total_distance = 0.0
    crime_weighted_sum = 0.0
    coordinates = []

    # Collect coordinates for the map
    for node in path:
        attrs = G.nodes[node]

        coordinates.append({
            "latitude": float(attrs["y"]),
            "longitude": float(attrs["x"])
        })

    # Calculate distance and crime exposure
    for u, v in zip(path, path[1:]):

        edges = G.get_edge_data(u, v)

        if not edges:
            continue

        # Select the parallel edge matching the route's
        # optimization criterion.
        def edge_cost(attrs):
            try:
                value = float(attrs.get(optimization, 1.0))

                if not math.isfinite(value):
                    return float("inf")

                return value
            except (TypeError, ValueError):
                return float("inf")

        edge = min(
            edges.values(),
            key=edge_cost
        )

        length = float(edge["length"])
        crime = float(edge["crime_index"])

        total_distance += length
        crime_weighted_sum += length * crime

    # Distance-weighted average crime index
    if total_distance > 0:
        average_crime = (
            crime_weighted_sum / total_distance
        )
    else:
        average_crime = float(df["Crime_Index"].mean())

    maximum_crime = float(df["Crime_Index"].max())

    # Relative safety score based on the dataset's maximum
    safety_score = round(
        max(
            0,
            min(
                100,
                (1 - average_crime / maximum_crime) * 100
            )
        )
    )

    # Categorize crime risk using dataset quantiles
    lower_threshold = float(
        df["Crime_Index"].quantile(0.33)
    )

    upper_threshold = float(
        df["Crime_Index"].quantile(0.67)
    )

    if average_crime < lower_threshold:
        crime_risk = "Low"
    elif average_crime < upper_threshold:
        crime_risk = "Medium"
    else:
        crime_risk = "High"

    # Estimated walking time at approximately 5 km/h
    # A positive route takes at least one estimated minute.
    if total_distance > 0:
        estimated_minutes = max(
            1,
            math.ceil(total_distance / 5000 * 60)
        )
    else:
        estimated_minutes = 0

    return {
        "distance_m": round(total_distance, 2),
        "distance_km": round(total_distance / 1000, 2),
        "estimated_time_minutes": estimated_minutes,
        "average_crime_index": round(average_crime, 2),
        "safety_score": safety_score,
        "crime_risk": crime_risk,
        "coordinates": coordinates
    }


# --------------------------------------------------
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