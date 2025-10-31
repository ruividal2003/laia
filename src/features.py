import numpy as np
import pandas as pd

def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Standard TLC columns vary slightly by year; normalize likely names:
    df = df.rename(columns={
        "tpep_pickup_datetime":"pickup_datetime",
        "tpep_dropoff_datetime":"dropoff_datetime",
        "PULocationID":"pu_zone",
        "DOLocationID":"do_zone"
    })

    df["duration_min"] = (pd.to_datetime(df["dropoff_datetime"]) -
                          pd.to_datetime(df["pickup_datetime"])).dt.total_seconds() / 60.0
    # quick sanity filter
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 180)]

    # time features
    ts = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = ts.dt.hour
    df["pickup_dow"]  = ts.dt.dayofweek
    df["pickup_mon"]  = ts.dt.month

    # geo features (drop rows with missing coords)
    for c in ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]:
        if c not in df.columns:
            # older files use 'pickup_longitude' style; newer are location IDs – we handle classic coords baseline
            df[c] = np.nan
    mask = df[["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]].notna().all(axis=1)
    df = df[mask].copy()
    df["hav_km"] = haversine(df["pickup_longitude"], df["pickup_latitude"],
                             df["dropoff_longitude"], df["dropoff_latitude"])

    # keep compact set
    cols = ["trip_distance","hav_km","passenger_count","pickup_hour","pickup_dow","pickup_mon","duration_min"]
    return df[cols].dropna()
