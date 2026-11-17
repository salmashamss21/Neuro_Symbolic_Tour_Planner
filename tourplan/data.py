"""Data loading, preprocessing, splits and sample builders for the Flickr tour data.

Real columns come from the tour-cikm16 repository (Lim et al.).  Everything that is
generated (travel time, local-hour correction, clipped duration, budgets, ...) is
listed in docs/DATA_DICTIONARY.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

RAW_URL = "https://raw.githubusercontent.com/computationalmedia/tour-cikm16/master/data/{name}"
REPO_URL = "https://github.com/computationalmedia/tour-cikm16.git"

CITIES = {"Edin": "Edinburgh", "Glas": "Glasgow", "Melb": "Melbourne",
          "Osak": "Osaka", "Toro": "Toronto"}

# ---- preprocessing constants (all documented in the data dictionary) -------------
MIN_VISITS = 20          # keep places with at least this many visits
MAX_PLACES = 60          # ... and at most this many (most visited first)
MIN_DUR_S = 300          # durations below 5 minutes are clipped up to 5 minutes
MAX_DUR_S = 4 * 3600     # very long "visits" (photo gaps) are clipped to 4 hours
# The stored timestamps are not local time: in all five cities photo activity peaks
# at the same stored hour (03-04h) and is lowest at 16-19h, i.e. a fixed offset was
# applied.  Shifting by +11h puts the night trough at ~05h and the peak at ~15h.
LOCAL_HOUR_OFFSET = 11.0
WALK_KMH = 4.5           # generated: walking speed
DETOUR = 1.3             # generated: street-network detour factor over straight line


# ------------------------------------------------------------------ download -----
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw(city: str = "Edin", force: bool = False, raw_dir: Path = RAW_DIR) -> None:
    """Download poi-<city>.csv / traj-<city>.csv if missing (HTTP, then git fallback)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = [f"poi-{city}.csv", f"traj-{city}.csv"]
    missing = [n for n in names if force or not (raw_dir / n).exists()]
    if not missing:
        return
    try:
        for n in missing:
            urllib.request.urlretrieve(RAW_URL.format(name=n), raw_dir / n)
        return
    except Exception as exc:  # noqa: BLE001 - network failure is expected in some sandboxes
        print(f"[data] HTTP download failed ({exc}); falling back to git clone")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, tmp + "/repo"], check=True)
        for n in missing:
            (raw_dir / n).write_bytes((Path(tmp) / "repo" / "data" / n).read_bytes())


def write_checksums(raw_dir: Path = RAW_DIR) -> dict:
    sums = {p.name: sha256(p) for p in sorted(raw_dir.glob("*.csv"))}
    (raw_dir / "CHECKSUMS.json").write_text(json.dumps(sums, indent=1))
    return sums


# ------------------------------------------------------------------- geometry -----
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def travel_matrix_min(lat: np.ndarray, lon: np.ndarray,
                      speed_kmh: float = WALK_KMH, detour: float = DETOUR) -> np.ndarray:
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return d * detour / speed_kmh * 60.0


# ---------------------------------------------------------------- city object -----
@dataclass
class CityData:
    city: str
    pois: pd.DataFrame            # index = poi_idx 0..P-1
    visits: pd.DataFrame          # one row per visit, sorted by (traj, start)
    categories: list              # sorted category names
    travel_min: np.ndarray        # P x P generated walking minutes
    meta: dict = field(default_factory=dict)

    @property
    def n_places(self) -> int:
        return len(self.pois)

    @property
    def cat_of(self) -> np.ndarray:
        """Category index of each place."""
        return self.pois["cat_idx"].to_numpy()

    def places_in(self, cat_idx: int) -> np.ndarray:
        return np.flatnonzero(self.cat_of == cat_idx)


def build_city(city: str = "Edin", raw_dir: Path = RAW_DIR,
               min_visits: int = MIN_VISITS, max_places: int = MAX_PLACES) -> CityData:
    poi = pd.read_csv(raw_dir / f"poi-{city}.csv")
    tr = pd.read_csv(raw_dir / f"traj-{city}.csv")
    n_raw_places, n_raw_visits = len(poi), len(tr)

    counts = tr["poiID"].value_counts()
    keep = counts[counts >= min_visits].sort_values(ascending=False, kind="stable").index[:max_places]
    keep = sorted(keep)
    poi = poi[poi["poiID"].isin(keep)].sort_values("poiID").reset_index(drop=True)
    poi["poi_idx"] = np.arange(len(poi))
    id2idx = dict(zip(poi["poiID"], poi["poi_idx"]))
    categories = sorted(poi["poiCat"].unique())
    poi["cat_idx"] = poi["poiCat"].map({c: i for i, c in enumerate(categories)})
    poi["n_visits"] = poi["poiID"].map(counts).astype(int)
    poi = poi.rename(columns={"poiID": "poi_id", "poiCat": "category",
                              "poiLon": "lon", "poiLat": "lat"})
    poi = poi[["poi_idx", "poi_id", "category", "cat_idx", "lon", "lat", "n_visits"]]

    tr = tr[tr["poiID"].isin(id2idx)].copy()
    tr = tr.sort_values(["trajID", "startTime", "endTime"], kind="stable")
    tr["poi_idx"] = tr["poiID"].map(id2idx).astype(int)
    # drop immediate repeats of the same place inside one trajectory (none in Edinburgh)
    prev = tr.groupby("trajID")["poi_idx"].shift()
    tr = tr[tr["poi_idx"] != prev].copy()
    tr["dur_clip_s"] = tr["poiDuration"].clip(MIN_DUR_S, MAX_DUR_S)
    tr["hour_start"] = ((tr["startTime"] % 86400) / 3600.0 + LOCAL_HOUR_OFFSET) % 24
    tr["hour_end"] = ((tr["endTime"] % 86400) / 3600.0 + LOCAL_HOUR_OFFSET) % 24
    tr["pos_in_traj"] = tr.groupby("trajID").cumcount()
    tr["traj_len"] = tr.groupby("trajID")["poi_idx"].transform("size")
    visits = tr.rename(columns={"userID": "user", "trajID": "traj", "startTime": "start_ts",
                                "endTime": "end_ts", "#photo": "n_photo",
                                "poiDuration": "duration_s"})
    visits = visits[["user", "traj", "poi_idx", "pos_in_traj", "traj_len", "start_ts", "end_ts",
                     "n_photo", "duration_s", "dur_clip_s", "hour_start", "hour_end"]].reset_index(drop=True)

    travel = travel_matrix_min(poi["lat"].to_numpy(), poi["lon"].to_numpy())
    meta = dict(city=city, n_raw_places=int(n_raw_places), n_raw_visits=int(n_raw_visits),
                n_places=int(len(poi)), n_visits=int(len(visits)),
                n_users=int(visits["user"].nunique()), n_trajs=int(visits["traj"].nunique()),
                n_transitions=int((visits["traj_len"] - 1).clip(lower=0)[visits["pos_in_traj"] == 0].sum()),
                min_visits=min_visits, max_places=max_places, min_dur_s=MIN_DUR_S,
                max_dur_s=MAX_DUR_S, local_hour_offset=LOCAL_HOUR_OFFSET,
                walk_kmh=WALK_KMH, detour=DETOUR)
    return CityData(city, poi, visits, categories, travel, meta)


def save_processed(cd: CityData, out_dir: Path = PROC_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cd.pois.to_csv(out_dir / f"{cd.city}_places.csv", index=False)
    cd.visits.to_csv(out_dir / f"{cd.city}_visits.csv", index=False)
    np.save(out_dir / f"{cd.city}_travel_min.npy", cd.travel_min)
    (out_dir / f"{cd.city}_meta.json").write_text(json.dumps(cd.meta, indent=1))


def load_processed(city: str = "Edin", out_dir: Path = PROC_DIR) -> CityData:
    pois = pd.read_csv(out_dir / f"{city}_places.csv")
    visits = pd.read_csv(out_dir / f"{city}_visits.csv")
    travel = np.load(out_dir / f"{city}_travel_min.npy")
    meta = json.loads((out_dir / f"{city}_meta.json").read_text())
    cats = sorted(pois["category"].unique())
    return CityData(city, pois, visits, cats, travel, meta)


def get_city(city: str = "Edin") -> CityData:
    """Processed data if available, otherwise build it from raw files."""
    if (PROC_DIR / f"{city}_visits.csv").exists():
        return load_processed(city)
    ensure_raw(city)
    cd = build_city(city)
    save_processed(cd)
    return cd


# ---------------------------------------------------------------------- splits ----
def split_users(visits: pd.DataFrame, seed: int, frac=(0.70, 0.15, 0.15)) -> dict:
    """Split USERS (not trajectories) into train/val/test so no user leaks across sets."""
    users = np.array(sorted(visits["user"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n = len(users)
    n_tr, n_va = int(round(frac[0] * n)), int(round(frac[1] * n))
    return {"train": set(users[:n_tr]), "val": set(users[n_tr:n_tr + n_va]),
            "test": set(users[n_tr + n_va:])}


def transitions(visits: pd.DataFrame, users: set | None = None):
    """(current_place, hour_leaving_current, next_place) for every consecutive visit pair."""
    v = visits if users is None else visits[visits["user"].isin(users)]
    v = v.sort_values(["traj", "pos_in_traj"])
    nxt = v.groupby("traj")["poi_idx"].shift(-1)
    m = nxt.notna()
    return (v.loc[m, "poi_idx"].to_numpy().astype(int),
            v.loc[m, "hour_end"].to_numpy().astype(float),
            nxt[m].to_numpy().astype(int))


def context_transitions(visits: pd.DataFrame, users: set | None = None):
    """(previous_place, current_place, hour_leaving_current, next_place).

    The first transition in a trajectory has previous_place=-1 and backs off to
    the first-order current-place model.
    """
    v = visits if users is None else visits[visits["user"].isin(users)]
    v = v.sort_values(["traj", "pos_in_traj"])
    group = v.groupby("traj")["poi_idx"]
    prev, nxt = group.shift(1), group.shift(-1)
    m = nxt.notna()
    return (prev[m].fillna(-1).to_numpy().astype(int),
            v.loc[m, "poi_idx"].to_numpy().astype(int),
            v.loc[m, "hour_end"].to_numpy().astype(float),
            nxt[m].to_numpy().astype(int))


def goal_transitions(visits: pd.DataFrame, users: set | None = None):
    """(current, hour, trajectory_end, stops_remaining, next) for each transition."""
    v = visits if users is None else visits[visits["user"].isin(users)]
    v = v.sort_values(["traj", "pos_in_traj"])
    groups = v.groupby("traj", sort=False)
    nxt = groups["poi_idx"].shift(-1)
    end = groups["poi_idx"].transform("last")
    lengths = groups["poi_idx"].transform("size")
    remaining = lengths - v["pos_in_traj"] - 1
    position = v["pos_in_traj"] / (lengths - 1).clip(lower=1)
    mask = nxt.notna()
    return (v.loc[mask, "poi_idx"].to_numpy(dtype=int),
            v.loc[mask, "hour_end"].to_numpy(dtype=float),
            end.loc[mask].to_numpy(dtype=int),
            remaining.loc[mask].to_numpy(dtype=float),
            position.loc[mask].to_numpy(dtype=float),
            nxt.loc[mask].to_numpy(dtype=int))


def duration_samples(visits: pd.DataFrame, users: set | None = None):
    """(place, arrival hour, log clipped duration in seconds) for every visit."""
    v = visits if users is None else visits[visits["user"].isin(users)]
    return (v["poi_idx"].to_numpy().astype(int), v["hour_start"].to_numpy().astype(float),
            np.log(v["dur_clip_s"].to_numpy().astype(float)))


def trajectories(visits: pd.DataFrame, users: set | None = None, min_len: int = 2):
    """List of dicts {traj, user, places[list], hours_start[list], dur_s[list]}."""
    v = visits if users is None else visits[visits["user"].isin(users)]
    out = []
    for traj, g in v.sort_values(["traj", "pos_in_traj"]).groupby("traj", sort=True):
        if len(g) >= min_len:
            out.append(dict(traj=int(traj), user=g["user"].iloc[0], places=g["poi_idx"].tolist(),
                            hours_start=g["hour_start"].tolist(), dur_s=g["dur_clip_s"].tolist()))
    return out
