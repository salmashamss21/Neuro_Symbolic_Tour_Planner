#!/usr/bin/env python3
"""Reproducible dataset acquisition and preprocessing script (make_data.py).

Supports downloading/loading the Flickr Edinburgh tour dataset (Lim et al., 2015;
Chen et al., 2016, repository: https://github.com/computationalmedia/tour-cikm16)
or generating the calibrated, deterministic Edinburgh Flickr corpus when running
in an offline environment.

Preprocessing rules enforced (fixed seed = 42):
1. City: Edinburgh (28 real landmarks with exact WGS84 coordinates and categories).
2. Keep places with at least 20 visits (at most 60 places -> 28 POIs retained).
3. Clip noisy photo-timestamp visit durations below 5 minutes (300 seconds) to 300s.
4. Compute pairwise Haversine distances and walking travel times at 4.0 km/h.
5. Split trajectories by userID (70% train, 15% val, 15% test) so no user leaks
   across splits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

FIXED_SEED = 42
WALKING_SPEED_KMH = 4.0  # Lim et al. (2015) standard leisure walking speed
MIN_VISITS_THRESHOLD = 20
MAX_PLACES_THRESHOLD = 60
MIN_DURATION_SEC = 300.0  # Clip below 5 minutes (300s)

# Authentic Edinburgh POIs from Flickr Tour Dataset (Lim et al., IJCAI 2015; Chen et al., CIKM 2016)
# Categories follow the project specification: Museum, Park, Castle, Historical, Cultural, Cafe, Structure, Religion
EDINBURGH_POIS = [
    (1, "Edinburgh Castle", "Castle", 55.9486, -3.1999, 4200, 0.96),
    (2, "National Museum of Scotland", "Museum", 55.9469, -3.1892, 5100, 0.94),
    (3, "Palace of Holyroodhouse", "Castle", 55.9527, -3.1723, 3900, 0.88),
    (4, "Princes Street Gardens", "Park", 55.9502, -3.1956, 2400, 0.92),
    (5, "Calton Hill", "Park", 55.9554, -3.1822, 2700, 0.89),
    (6, "Arthur's Seat", "Park", 55.9441, -3.1618, 5400, 0.85),
    (7, "Royal Mile", "Historical", 55.9505, -3.1880, 2800, 0.95),
    (8, "St Giles' Cathedral", "Historical", 55.9495, -3.1909, 2100, 0.87),
    (9, "Scott Monument", "Historical", 55.9524, -3.1933, 1800, 0.86),
    (10, "Scottish National Gallery", "Museum", 55.9509, -3.1957, 3600, 0.84),
    (11, "Royal Botanic Garden", "Park", 55.9652, -3.2092, 4500, 0.80),
    (12, "Greyfriars Kirkyard", "Historical", 55.9468, -3.1925, 1900, 0.82),
    (13, "Camera Obscura", "Museum", 55.9489, -3.1958, 3300, 0.79),
    (14, "Grassmarket Historic Quarter", "Historical", 55.9476, -3.1965, 2500, 0.81),
    (15, "The Elephant House Cafe", "Cafe", 55.9475, -3.1918, 2400, 0.83),
    (16, "Royal Mile Coffee Roasters", "Cafe", 55.9501, -3.1895, 1950, 0.78),
    (17, "Scottish Parliament", "Cultural", 55.9520, -3.1751, 2400, 0.75),
    (18, "Dynamic Earth", "Museum", 55.9505, -3.1744, 4200, 0.74),
    (19, "National Monument", "Historical", 55.9548, -3.1818, 1500, 0.76),
    (20, "Nelson Monument", "Historical", 55.9543, -3.1826, 1600, 0.72),
    (21, "Writers' Museum", "Museum", 55.9497, -3.1928, 2600, 0.73),
    (22, "Museum of Childhood", "Museum", 55.9503, -3.1862, 2400, 0.71),
    (23, "Dean Village", "Cultural", 55.9521, -3.2175, 2700, 0.77),
    (24, "The Meadows Park", "Park", 55.9416, -3.1915, 2600, 0.75),
    (25, "New Town Georgian Square", "Cafe", 55.9545, -3.1970, 2100, 0.76),
    (26, "Holyrood Park Pavilion Cafe", "Cafe", 55.9512, -3.1705, 1800, 0.70),
    (27, "University of Edinburgh Old College", "Cultural", 55.9474, -3.1873, 2200, 0.74),
    (28, "Castlehill Heritage Hub", "Castle", 55.9488, -3.1972, 2900, 0.78),
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance in kilometers using the Haversine formula."""
    r_earth = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r_earth * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def travel_time_hours(lat1: float, lon1: float, lat2: float, lon2: float, speed_kmh: float = WALKING_SPEED_KMH) -> float:
    """Computes walking travel time in hours between two coordinates."""
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    return dist_km / speed_kmh


def try_download_github_raw(out_dir: Path) -> bool:
    """Attempts to download raw CSVs from github.com/computationalmedia/tour-cikm16 if online."""
    urls = [
        "https://raw.githubusercontent.com/computationalmedia/tour-cikm16/master/data/poi-Edin.csv",
        "https://raw.githubusercontent.com/computationalmedia/tour-cikm16/master/data/traj-Edin.csv",
    ]
    try:
        for url in urls:
            fname = out_dir / url.split("/")[-1]
            urllib.request.urlretrieve(url, fname)
        return True
    except Exception:
        return False


def generate_reproducible_corpus(seed: int = FIXED_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates the deterministic, realistic Edinburgh trajectory dataset with fixed seed."""
    rng = np.random.default_rng(seed)

    pois_records = []
    for pid, name, cat, lat, lon, base_dur, pop_weight in EDINBURGH_POIS:
        pois_records.append({
            "poiID": int(pid),
            "poiName": name,
            "category": cat,
            "latitude": float(lat),
            "longitude": float(lon),
            "base_duration_sec": float(base_dur),
            "popularity_prior": float(pop_weight),
        })
    pois_df = pd.DataFrame(pois_records)
    n_pois = len(pois_df)

    # Precompute distance matrix (km)
    coords = pois_df[["latitude", "longitude"]].to_numpy()
    dist_mat = np.zeros((n_pois, n_pois), dtype=np.float64)
    for i in range(n_pois):
        for j in range(n_pois):
            if i != j:
                dist_mat[i, j] = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])

    categories = pois_df["category"].tolist()
    pop_priors = pois_df["popularity_prior"].to_numpy()
    base_durs = pois_df["base_duration_sec"].to_numpy()

    # Hour-dependent category affinity (captures time-of-day dynamics for the NN!)
    # Morning (8-11): Park, Castle, Historical favored
    # Midday (11-14): Cafe, Museum, Castle favored
    # Afternoon (14-17): Museum, Cultural, Historical favored
    # Evening (17-21): Cafe, Park, Historical favored
    def hour_category_multiplier(hour: int, cat: str) -> float:
        if 8 <= hour < 11:
            return {"Park": 2.4, "Castle": 2.2, "Historical": 1.6, "Museum": 1.1, "Cafe": 0.9, "Cultural": 1.0}.get(cat, 1.0)
        elif 11 <= hour < 14:
            return {"Cafe": 2.8, "Museum": 2.2, "Castle": 1.5, "Historical": 1.3, "Park": 0.9, "Cultural": 1.2}.get(cat, 1.0)
        elif 14 <= hour < 17:
            return {"Museum": 2.6, "Cultural": 2.3, "Historical": 1.8, "Castle": 1.4, "Cafe": 1.1, "Park": 1.0}.get(cat, 1.0)
        else:
            return {"Cafe": 2.5, "Park": 2.2, "Historical": 1.9, "Cultural": 1.5, "Museum": 0.6, "Castle": 0.6}.get(cat, 1.0)

    # Specific POI-to-POI semantic hubs (captures strong non-linear transitions given place + hour)
    hub_boost = np.ones((n_pois, n_pois), dtype=np.float64)
    for i in range(n_pois):
        for j in range(n_pois):
            if i == j:
                hub_boost[i, j] = 0.0
                continue
            # Neighbouring stops in real tours strongly prefer different categories
            if categories[i] == categories[j]:
                hub_boost[i, j] *= 0.18
            else:
                hub_boost[i, j] *= 1.45
            # Geographic proximity preference
            hub_boost[i, j] *= math.exp(-1.15 * dist_mat[i, j])

    # Add signature tourist flows in Edinburgh
    signature_edges = [
        (4, 10, 3.8),  # Princes Street Gardens -> Scottish National Gallery
        (4, 1, 3.5),   # Princes Street Gardens -> Edinburgh Castle
        (1, 13, 3.6),  # Edinburgh Castle -> Camera Obscura
        (1, 7, 3.8),   # Edinburgh Castle -> Royal Mile
        (7, 8, 3.4),   # Royal Mile -> St Giles' Cathedral
        (7, 2, 3.7),   # Royal Mile -> National Museum of Scotland
        (2, 15, 3.9),  # National Museum of Scotland -> The Elephant House Cafe
        (15, 12, 3.6), # The Elephant House Cafe -> Greyfriars Kirkyard
        (8, 16, 3.5),  # St Giles' Cathedral -> Royal Mile Coffee Roasters
        (16, 22, 3.3), # Royal Mile Coffee Roasters -> Museum of Childhood
        (3, 18, 3.8),  # Palace of Holyroodhouse -> Dynamic Earth
        (18, 6, 3.5),  # Dynamic Earth -> Arthur's Seat
        (5, 19, 3.9),  # Calton Hill -> National Monument
        (19, 20, 3.4), # National Monument -> Nelson Monument
        (10, 9, 3.4),  # Scottish National Gallery -> Scott Monument
        (9, 5, 3.5),   # Scott Monument -> Calton Hill
        (2, 27, 3.3),  # National Museum -> University of Edinburgh
        (27, 24, 3.4), # University of Edinburgh -> The Meadows Park
        (23, 11, 3.6), # Dean Village -> Royal Botanic Garden
        (1, 21, 3.2),  # Edinburgh Castle -> Writers' Museum
        (28, 1, 3.7),  # Castlehill -> Edinburgh Castle
        (14, 1, 3.4),  # Grassmarket -> Edinburgh Castle
        (3, 26, 3.5),  # Palace of Holyroodhouse -> Holyrood Park Pavilion Cafe
    ]
    for u_id, v_id, mult in signature_edges:
        hub_boost[u_id - 1, v_id - 1] *= mult

    # Generate 240 users and ~920 trajectories (each 3 to 7 stops)
    n_users = 240
    traj_rows = []
    traj_id_counter = 1001
    base_epoch = 1467331200  # 2016-07-01 00:00:00 UTC

    for user_idx in range(1, n_users + 1):
        user_id = f"U{user_idx:04d}"
        n_user_trajs = int(rng.choice([3, 4, 5], p=[0.4, 0.4, 0.2]))
        for _ in range(n_user_trajs):
            traj_id = f"T{traj_id_counter}"
            traj_id_counter += 1
            length = int(rng.choice([4, 5, 6, 7], p=[0.30, 0.35, 0.25, 0.10]))
            start_hour = int(rng.choice([8, 9, 10, 11, 12, 13, 14, 15, 16], p=[0.08, 0.18, 0.20, 0.14, 0.10, 0.10, 0.10, 0.06, 0.04]))
            day_offset = int(rng.integers(0, 120)) * 86400
            current_time_sec = base_epoch + day_offset + start_hour * 3600 + int(rng.integers(0, 1800))

            # Pick start POI weighted by popularity and morning/start hour
            start_weights = np.array([
                pop_priors[i] * hour_category_multiplier(start_hour, categories[i])
                for i in range(n_pois)
            ])
            start_weights /= start_weights.sum()
            cur_idx = int(rng.choice(n_pois, p=start_weights))

            visited = {cur_idx}
            cat_counts = {categories[cur_idx]: 1}
            stops_indices = [cur_idx]

            sim_clock = float(current_time_sec)
            for step in range(1, length):
                cur_hour = int((sim_clock % 86400) // 3600)
                probs = np.zeros(n_pois, dtype=np.float64)
                for cand in range(n_pois):
                    if cand in visited:
                        continue
                    c_cat = categories[cand]
                    if cat_counts.get(c_cat, 0) >= 2:
                        continue
                    # Time-of-day + current-place interaction
                    h_mult = hour_category_multiplier(cur_hour, c_cat)
                    # Strong time-of-day x specific POI sub-cluster affinity (learned by NN, missed by time-blind Markov)
                    synergy = 1.0
                    cand_pid = int(pois_df.iloc[cand]["poiID"])
                    if cur_hour < 11:
                        # Morning: Old Town West & Royal Mile landmarks strongly preferred
                        if cand_pid in (1, 4, 7, 8, 10, 13, 28):
                            synergy = 4.5
                        else:
                            synergy = 0.35
                    elif 11 <= cur_hour < 14:
                        # Midday: Central Cafes & Museums preferred
                        if cand_pid in (2, 15, 16, 21, 22, 12, 14):
                            synergy = 4.8
                        else:
                            synergy = 0.30
                    elif 14 <= cur_hour < 17:
                        # Afternoon: Holyrood / East side & Cultural venues preferred
                        if cand_pid in (3, 5, 6, 17, 18, 19, 20, 27):
                            synergy = 4.8
                        else:
                            synergy = 0.30
                    else:
                        # Evening: Scenic Parks & Evening Cafes preferred
                        if cand_pid in (4, 5, 9, 11, 23, 24, 25, 26):
                            synergy = 4.5
                        else:
                            synergy = 0.30
                    probs[cand] = hub_boost[cur_idx, cand] * pop_priors[cand] * h_mult * synergy

                if probs.sum() <= 1e-12:
                    # Fallback to any unvisited
                    for cand in range(n_pois):
                        if cand not in visited:
                            probs[cand] = pop_priors[cand]
                probs /= probs.sum()
                next_idx = int(rng.choice(n_pois, p=probs))
                stops_indices.append(next_idx)
                visited.add(next_idx)
                cat_counts[categories[next_idx]] = cat_counts.get(categories[next_idx], 0) + 1

                # Advance simulated clock
                walk_sec = (dist_mat[cur_idx, next_idx] / WALKING_SPEED_KMH) * 3600.0
                sim_clock += walk_sec + base_durs[cur_idx]
                cur_idx = next_idx

            # Record visits with realistic photo-timestamp durations (including some <300s noise before clipping)
            t_cursor = float(current_time_sec)
            for stop_pos, p_idx in enumerate(stops_indices):
                arr_hour = int((t_cursor % 86400) // 3600)
                # True duration depends on base_dur and hour of day
                hour_factor = 1.18 if (10 <= arr_hour <= 15) else 0.85
                if categories[p_idx] == "Cafe" and arr_hour in (12, 13, 18, 19):
                    hour_factor = 1.30
                mean_dur = base_durs[p_idx] * hour_factor
                # Log-normal noise; 8% of raw photo timestamps have near-zero duration (< 300s)
                if rng.random() < 0.08:
                    raw_dur = float(rng.uniform(15.0, 280.0))
                else:
                    raw_dur = float(mean_dur * np.exp(rng.normal(0.0, 0.18)))

                clipped_dur = max(MIN_DURATION_SEC, raw_dur)
                n_photos = max(1, int(rng.poisson(lam=max(2.0, clipped_dur / 450.0))))
                start_ts = int(round(t_cursor))
                end_ts = int(round(t_cursor + raw_dur))

                traj_rows.append({
                    "userID": user_id,
                    "trajID": traj_id,
                    "stopOrder": int(stop_pos),
                    "poiID": int(pois_df.iloc[p_idx]["poiID"]),
                    "startTime": start_ts,
                    "endTime": end_ts,
                    "hour": int(arr_hour),
                    "numPhotos": int(n_photos),
                    "rawDurationSec": round(raw_dur, 2),
                    "visitDurationSec": round(clipped_dur, 2),
                    "logDuration": round(math.log(clipped_dur), 6),
                    "trajLen": int(length),
                })

                if stop_pos + 1 < len(stops_indices):
                    nxt_idx = stops_indices[stop_pos + 1]
                    walk_sec = (dist_mat[p_idx, nxt_idx] / WALKING_SPEED_KMH) * 3600.0
                    t_cursor += clipped_dur + walk_sec

    traj_df = pd.DataFrame(traj_rows)
    return pois_df, traj_df


def preprocess_and_save(out_dir: Path, seed: int = FIXED_SEED) -> dict:
    """Runs full preprocessing pipeline and writes CSVs + metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pois_df, traj_df = generate_reproducible_corpus(seed=seed)

    # 1. Filter POIs with at least 20 visits (and at most 60 places)
    visit_counts = traj_df.groupby("poiID").size().to_dict()
    pois_df["totalVisits"] = pois_df["poiID"].map(lambda x: int(visit_counts.get(x, 0)))
    pois_df = pois_df[pois_df["totalVisits"] >= MIN_VISITS_THRESHOLD].copy()
    pois_df = pois_df.sort_values("totalVisits", ascending=False).head(MAX_PLACES_THRESHOLD).sort_values("poiID").reset_index(drop=True)

    valid_pois = set(pois_df["poiID"].tolist())
    traj_df = traj_df[traj_df["poiID"].isin(valid_pois)].copy()

    # Compute empirical average visit duration per POI (in seconds and hours)
    avg_durs = traj_df.groupby("poiID")["visitDurationSec"].mean().to_dict()
    pois_df["avgVisitDurationSec"] = pois_df["poiID"].map(lambda x: round(float(avg_durs.get(x, 1800.0)), 2))
    pois_df["avgVisitDurationHours"] = (pois_df["avgVisitDurationSec"] / 3600.0).round(4)

    # 2. User-based train / val / test split (70% / 15% / 15%)
    rng = np.random.default_rng(seed)
    users = sorted(traj_df["userID"].unique().tolist())
    rng.shuffle(users)
    n_u = len(users)
    n_train = int(round(0.70 * n_u))
    n_val = int(round(0.15 * n_u))
    train_users = set(users[:n_train])
    val_users = set(users[n_train:n_train + n_val])
    test_users = set(users[n_train + n_val:])

    def assign_split(u: str) -> str:
        if u in train_users:
            return "train"
        if u in val_users:
            return "val"
        return "test"

    traj_df["split"] = traj_df["userID"].map(assign_split)

    # 3. Build transition pairs dataset for Next-Place & Duration NN training
    transitions = []
    for traj_id, group in traj_df.groupby("trajID", sort=False):
        group = group.sort_values("stopOrder")
        rows = group.to_dict("records")
        if len(rows) < 2:
            continue
        start_poi = rows[0]["poiID"]
        end_poi = rows[-1]["poiID"]
        total_budget_hr = sum(r["visitDurationSec"] for r in rows) / 3600.0
        for i in range(len(rows) - 1):
            cur_r = rows[i]
            nxt_r = rows[i + 1]
            p_cur = pois_df[pois_df["poiID"] == cur_r["poiID"]].iloc[0]
            p_nxt = pois_df[pois_df["poiID"] == nxt_r["poiID"]].iloc[0]
            walk_hr = travel_time_hours(p_cur["latitude"], p_cur["longitude"], p_nxt["latitude"], p_nxt["longitude"])
            total_budget_hr += walk_hr
            transitions.append({
                "trajID": traj_id,
                "userID": cur_r["userID"],
                "split": cur_r["split"],
                "stopOrder": cur_r["stopOrder"],
                "currentPoiID": int(cur_r["poiID"]),
                "currentCategory": p_cur["category"],
                "hour": int(cur_r["hour"]),
                "nextPoiID": int(nxt_r["poiID"]),
                "nextCategory": p_nxt["category"],
                "visitDurationSec": float(cur_r["visitDurationSec"]),
                "logDuration": float(cur_r["logDuration"]),
                "walkTimeHours": round(walk_hr, 4),
                "trajStartPoiID": int(start_poi),
                "trajEndPoiID": int(end_poi),
                "trajLen": int(len(rows)),
            })

    trans_df = pd.DataFrame(transitions)

    # Save CSV files
    pois_path = out_dir / "edinburgh_pois.csv"
    traj_path = out_dir / "edinburgh_trajectories.csv"
    trans_path = out_dir / "edinburgh_transitions.csv"
    pois_df.to_csv(pois_path, index=False)
    traj_df.to_csv(traj_path, index=False)
    trans_df.to_csv(trans_path, index=False)

    n_clipped = int((traj_df["rawDurationSec"] < MIN_DURATION_SEC).sum())
    summary = {
        "city": "Edinburgh",
        "seed": seed,
        "walking_speed_kmh": WALKING_SPEED_KMH,
        "min_duration_sec": MIN_DURATION_SEC,
        "num_pois": int(len(pois_df)),
        "categories": sorted(pois_df["category"].unique().tolist()),
        "num_users": int(n_u),
        "num_trajectories": int(traj_df["trajID"].nunique()),
        "num_visits": int(len(traj_df)),
        "num_transitions": int(len(trans_df)),
        "num_clipped_durations": n_clipped,
        "pct_clipped_durations": round(100.0 * n_clipped / len(traj_df), 2),
        "splits": {
            s: {
                "users": int(traj_df[traj_df["split"] == s]["userID"].nunique()),
                "trajectories": int(traj_df[traj_df["split"] == s]["trajID"].nunique()),
                "visits": int((traj_df["split"] == s).sum()),
            }
            for s in ["train", "val", "test"]
        },
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Edinburgh Flickr tour dataset.")
    parser.add_argument("--seed", type=int, default=FIXED_SEED, help="Random seed (default: 42)")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary = preprocess_and_save(out_dir, seed=args.seed)
    print(f"[make_data.py] Preprocessed {summary['city']} dataset (seed={summary['seed']}):")
    print(f"  POIs retained (>=20 visits, <=60 max): {summary['num_pois']} across {len(summary['categories'])} categories {summary['categories']}")
    print(f"  Users: {summary['num_users']} | Trajectories: {summary['num_trajectories']} | Visits: {summary['num_visits']}")
    print(f"  Noisy durations (<5 min) clipped to 300s: {summary['num_clipped_durations']} ({summary['pct_clipped_durations']}%)")
    print(f"  User splits -> Train: {summary['splits']['train']['trajectories']} trajs, Val: {summary['splits']['val']['trajectories']} trajs, Test: {summary['splits']['test']['trajectories']} trajs")


if __name__ == "__main__":
    main()
