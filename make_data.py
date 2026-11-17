#!/usr/bin/env python
"""Download (if needed) and preprocess the Flickr tour data reproducibly.

    python make_data.py                # Edinburgh (default)
    python make_data.py --city Toro    # optional second city
    python make_data.py --force-download

Outputs data/processed/<city>_{places,visits}.csv, <city>_travel_min.npy, <city>_meta.json,
<city>_splits_seed<seed>.json (user split, fixed seed) and data/raw/CHECKSUMS.json.
"""
import argparse
import json
import random

import numpy as np

from tourplan import data as D

SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Edin", choices=list(D.CITIES))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force-download", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)

    D.ensure_raw(a.city, force=a.force_download)
    sums = D.write_checksums()
    cd = D.build_city(a.city)
    D.save_processed(cd)

    sp = D.split_users(cd.visits, a.seed)
    (D.PROC_DIR / f"{a.city}_splits_seed{a.seed}.json").write_text(
        json.dumps({k: sorted(v) for k, v in sp.items()}))
    print(json.dumps(cd.meta, indent=1))
    print("categories:", {c: int((cd.pois.category == c).sum()) for c in cd.categories})
    for k, u in sp.items():
        c, h, n = D.transitions(cd.visits, u)
        print(f"split {k:5s}: {len(u):4d} users, {len(c):5d} transitions")
    print("raw checksums written for", len(sums), "files")


if __name__ == "__main__":
    main()
