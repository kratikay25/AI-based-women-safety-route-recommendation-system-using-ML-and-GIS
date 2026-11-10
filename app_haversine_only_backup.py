def nearest_node(latitude, longitude, max_distance_m=1000):
    """
    Find the nearest graph node using Haversine distance.
    Reject locations more than 1 km from the road network.
    """

    best_node = None
    best_distance = float("inf")

    earth_radius = 6371000

    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)

    for node, attrs in G.nodes(data=True):
        try:
            lat = float(attrs["y"])
            lon = float(attrs["x"])
        except (KeyError, TypeError, ValueError):
            continue

        lat2 = math.radians(lat)
        lon2 = math.radians(lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        a = max(0.0, min(1.0, a))

        distance = 2 * earth_radius * math.asin(
            math.sqrt(a)
        )

        if distance < best_distance:
            best_distance = distance
            best_node = node

    if best_node is None:
        raise ValueError("No usable road-network nodes were found.")

    if best_distance > max_distance_m:
        raise ValueError(
            f"Selected location is {best_distance:.0f} metres "
            "from the available road network. "
            "Please select a location within 1 km."
        )

    return best_node
