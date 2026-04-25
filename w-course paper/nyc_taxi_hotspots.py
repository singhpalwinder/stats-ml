"""
NYC Yellow Taxi pickup hotspots for December 2025.

Pipeline:
  1. Load parquet with polars lazy frame, aggregate pickups per zone.
  2. Download/cache NYC TLC taxi zones shapefile, compute zone centroids.
  3. Render NYC street map base with osmnx (same style as maptoposter).
  4. Overlay hotspots as scatter points sized by pickup volume.
  5. Cluster zone centroids with sklearn HDBSCAN (weighted by pickup count
     via point replication) and color points by discovered cluster.
"""

import io
import os
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import polars as pl
import requests
from sklearn.cluster import HDBSCAN

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET = Path("/Users/paul/Downloads/yellow_tripdata_2025-12.parquet")
CACHE_DIR = SCRIPT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
ZONES_SHP = CACHE_DIR / "taxi_zones" / "taxi_zones" / "taxi_zones.shp"
OUT_PNG = SCRIPT_DIR / "nyc_taxi_hotspots_dec2025.png"

ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

# Map bounds (Manhattan + close boroughs). Center is roughly midtown Manhattan.
CENTER = (40.7580, -73.9855)
DIST_METERS = 14000


def load_zones() -> gpd.GeoDataFrame:
    """Download taxi zones shapefile if missing, return centroids in WGS84."""
    if not ZONES_SHP.exists():
        print("Downloading NYC taxi zones shapefile...")
        r = requests.get(ZONES_URL, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(CACHE_DIR / "taxi_zones")
        print(f"  extracted to {CACHE_DIR / 'taxi_zones'}")

    zones = gpd.read_file(ZONES_SHP)
    # Compute centroids in a projected CRS, then convert to lat/lon.
    zones_proj = zones.to_crs(epsg=2263)
    centroids = zones_proj.geometry.centroid.to_crs(epsg=4326)
    zones["lon"] = centroids.x.values
    zones["lat"] = centroids.y.values
    return zones[["LocationID", "zone", "borough", "lat", "lon"]]


def load_pickup_counts() -> pl.DataFrame:
    """Polars lazy aggregate of pickup counts per zone for December 2025."""
    lf = pl.scan_parquet(PARQUET)
    counts = (
        lf.filter(
            (pl.col("tpep_pickup_datetime") >= pl.datetime(2025, 12, 1))
            & (pl.col("tpep_pickup_datetime") < pl.datetime(2026, 1, 1))
        )
        .group_by("PULocationID")
        .agg(pl.len().alias("pickups"))
        .collect()
    )
    print(f"  aggregated {counts['pickups'].sum():,} pickups across "
          f"{counts.height} zones")
    return counts


def cluster_hotspots(df: gpd.GeoDataFrame) -> np.ndarray:
    """Run HDBSCAN on zone centroids weighted by pickup count.

    sklearn HDBSCAN has no sample_weight, so we replicate each centroid
    proportionally to its pickup volume, cluster the expanded set, and
    map labels back via nearest-unique-point (trivial since originals are
    preserved at the head of the array).
    """
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    counts = df["pickups"].to_numpy()

    # Scale replication so heavy zones get more weight but set stays manageable.
    reps = np.clip((counts / counts.max() * 50).astype(int), 1, 50)
    expanded = np.repeat(np.column_stack([lats, lons]), reps, axis=0)

    # Tiny jitter so repeated identical points don't break the density tree.
    rng = np.random.default_rng(42)
    expanded = expanded + rng.normal(0, 1e-4, expanded.shape)

    model = HDBSCAN(min_cluster_size=80, min_samples=10, cluster_selection_epsilon=0.01)
    labels_expanded = model.fit_predict(expanded)

    # First `reps[i]` entries belong to zone i — take the modal label per zone.
    out = np.empty(len(df), dtype=int)
    idx = 0
    for i, r in enumerate(reps):
        chunk = labels_expanded[idx:idx + r]
        # modal label (ignoring noise if a cluster label exists)
        vals, cnt = np.unique(chunk, return_counts=True)
        out[i] = vals[np.argmax(cnt)]
        idx += r
    n_clusters = len({l for l in out if l != -1})
    print(f"  HDBSCAN found {n_clusters} hotspot clusters "
          f"({(out == -1).sum()} noise zones)")
    return out


def render_poster(df: gpd.GeoDataFrame, labels: np.ndarray) -> None:
    print("Downloading street network (this takes a minute)...")
    G = ox.graph_from_point(
        CENTER, dist=DIST_METERS, dist_type="bbox", network_type="drive"
    )
    print(f"  {len(G.edges):,} street edges")

    print("Downloading water & parks...")
    try:
        water = ox.features_from_point(
            CENTER, tags={"natural": "water"}, dist=DIST_METERS
        )
    except Exception:
        water = None
    try:
        parks = ox.features_from_point(
            CENTER, tags={"leisure": "park"}, dist=DIST_METERS
        )
    except Exception:
        parks = None

    # Dark theme
    bg = "#0B0E14"
    road_color = "#2A3340"
    water_color = "#0F2030"
    park_color = "#0F1A12"

    fig, ax = plt.subplots(figsize=(12, 16), facecolor=bg)
    ax.set_facecolor(bg)

    if water is not None and not water.empty:
        water.plot(ax=ax, facecolor=water_color, edgecolor="none", zorder=1)
    if parks is not None and not parks.empty:
        parks.plot(ax=ax, facecolor=park_color, edgecolor="none", zorder=2)

    ox.plot_graph(
        G, ax=ax, bgcolor=bg, node_size=0,
        edge_color=road_color, edge_linewidth=0.4,
        show=False, close=False,
    )

    # --- Hotspot overlay ---
    # Size scales with sqrt(pickups) so top zones don't swallow the map.
    pickups = df["pickups"].to_numpy()
    sizes = np.sqrt(pickups) / np.sqrt(pickups.max()) * 1200 + 10

    # Color by HDBSCAN cluster; noise = dim grey.
    unique_labels = sorted(set(labels))
    palette = plt.get_cmap("plasma")(
        np.linspace(0.15, 0.95, max(len([l for l in unique_labels if l != -1]), 1))
    )
    color_map = {}
    ci = 0
    for lab in unique_labels:
        if lab == -1:
            color_map[lab] = mcolors.to_rgba("#555555", alpha=0.35)
        else:
            color_map[lab] = palette[ci]
            ci += 1
    colors = [color_map[l] for l in labels]

    ax.scatter(
        df["lon"], df["lat"],
        s=sizes, c=colors,
        edgecolor="white", linewidth=0.3,
        alpha=0.85, zorder=5,
    )

    # Lock viewport to the map radius (osmnx sometimes expands it).
    # Rough degree conversions at NYC latitude.
    lat0, lon0 = CENTER
    dlat = DIST_METERS / 111000
    dlon = DIST_METERS / (111000 * np.cos(np.radians(lat0)))
    ax.set_xlim(lon0 - dlon, lon0 + dlon)
    ax.set_ylim(lat0 - dlat, lat0 + dlat)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Title / attribution
    ax.text(
        0.5, 0.96,
        "NYC YELLOW TAXI PICKUP HOTSPOTS",
        transform=ax.transAxes, ha="center", va="top",
        color="white", fontsize=22, fontweight="bold", zorder=11,
    )
    ax.text(
        0.5, 0.93,
        "December 2025  ·  HDBSCAN clusters over zone centroids",
        transform=ax.transAxes, ha="center", va="top",
        color="#AAAAAA", fontsize=12, zorder=11,
    )
    ax.text(
        0.98, 0.02,
        "© OpenStreetMap contributors  ·  Data: NYC TLC",
        transform=ax.transAxes, ha="right", va="bottom",
        color="#888888", fontsize=8, zorder=11,
    )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(OUT_PNG, dpi=250, facecolor=bg, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {OUT_PNG}")


def main() -> None:
    print("Loading pickup counts (polars lazy)...")
    counts = load_pickup_counts()

    print("Loading taxi zones...")
    zones = load_zones()

    counts_dict = dict(zip(counts["PULocationID"].to_list(),
                            counts["pickups"].to_list()))
    zones["pickups"] = zones["LocationID"].map(counts_dict)
    merged = zones.dropna(subset=["pickups"]).copy()
    merged["pickups"] = merged["pickups"].astype(int)
    # Drop obviously bad rows (zero counts, missing coords)
    merged = merged[merged["pickups"] > 0].copy()
    # Keep only boroughs within the map radius — others would be off-canvas.
    merged = merged[merged["borough"].isin(
        ["Manhattan", "Brooklyn", "Queens", "Bronx"]
    )].copy()
    print(f"  {len(merged)} zones with data after filtering")

    print("Clustering with HDBSCAN...")
    labels = cluster_hotspots(merged)

    print("Rendering poster...")
    render_poster(merged, labels)


if __name__ == "__main__":
    main()
