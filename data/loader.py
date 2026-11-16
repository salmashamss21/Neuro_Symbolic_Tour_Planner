"""Data loader and geographical distance utilities for Edinburgh tour planning."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

WALKING_SPEED_KMH = 4.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes Haversine distance in km between two WGS84 coordinates."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


class TourDataset:
    """Encapsulates POIs, distance/travel-time matrices, and train/val/test trajectories."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.pois_df = pd.read_csv(self.data_dir / "edinburgh_pois.csv")
        self.traj_df = pd.read_csv(self.data_dir / "edinburgh_trajectories.csv")
        self.trans_df = pd.read_csv(self.data_dir / "edinburgh_transitions.csv")
        with open(self.data_dir / "dataset_summary.json", "r", encoding="utf-8") as f:
            self.summary = json.load(f)

        self.poi_ids = self.pois_df["poiID"].astype(int).tolist()
        self.poi_to_idx = {pid: i for i, pid in enumerate(self.poi_ids)}
        self.idx_to_poi = {i: pid for i, pid in enumerate(self.poi_ids)}
        self.poi_names = dict(zip(self.pois_df["poiID"].astype(int), self.pois_df["poiName"]))
        self.poi_categories = dict(zip(self.pois_df["poiID"].astype(int), self.pois_df["category"]))
        self.poi_coords = {
            int(row["poiID"]): (float(row["latitude"]), float(row["longitude"]))
            for _, row in self.pois_df.iterrows()
        }
        self.categories = sorted(list(set(self.poi_categories.values())))

        # Category expected visit durations in hours (25th-percentile feasible duration per category
        # so CSP expected time acts as a realistic filter without falsely rejecting compact POIs)
        cat_durs = self.traj_df[self.traj_df["split"] == "train"].copy()
        cat_durs["category"] = cat_durs["poiID"].map(self.poi_categories)
        self.category_mean_duration_hr = (
            cat_durs.groupby("category")["visitDurationSec"].quantile(0.25) / 3600.0
        ).to_dict()
        self.poi_mean_duration_hr = dict(
            zip(self.pois_df["poiID"].astype(int), self.pois_df["avgVisitDurationHours"].astype(float))
        )

        # Pairwise walking travel time in hours (at 4.0 km/h)
        self.travel_time_hr = {}
        neighbor_walk_times = []
        for u in self.poi_ids:
            lat1, lon1 = self.poi_coords[u]
            dists_u = []
            for v in self.poi_ids:
                if u == v:
                    self.travel_time_hr[(u, v)] = 0.0
                else:
                    lat2, lon2 = self.poi_coords[v]
                    t_hr = haversine_km(lat1, lon1, lat2, lon2) / WALKING_SPEED_KMH
                    self.travel_time_hr[(u, v)] = t_hr
                    dists_u.append(t_hr)
            neighbor_walk_times.extend(sorted(dists_u)[:4])
        self.mean_walk_time_hr = float(np.mean(neighbor_walk_times))
        self.min_walk_time_hr = float(np.min(neighbor_walk_times))

    def get_trajectories_by_split(self, split: str = "test") -> list[dict]:
        """Returns list of trajectory dicts for the given split."""
        sub = self.traj_df[self.traj_df["split"] == split]
        out = []
        for traj_id, grp in sub.groupby("trajID", sort=False):
            grp = grp.sort_values("stopOrder")
            pois = grp["poiID"].astype(int).tolist()
            durs_hr = (grp["visitDurationSec"].astype(float) / 3600.0).tolist()
            hours = grp["hour"].astype(int).tolist()
            cats = [self.poi_categories[p] for p in pois]
            walk_hr = sum(self.travel_time_hr[(pois[i], pois[i + 1])] for i in range(len(pois) - 1))
            total_hr = sum(durs_hr) + walk_hr
            out.append({
                "trajID": traj_id,
                "userID": grp["userID"].iloc[0],
                "pois": pois,
                "categories": cats,
                "durations_hr": durs_hr,
                "start_hour": hours[0],
                "total_time_hr": total_hr,
                "length": len(pois),
            })
        return out
