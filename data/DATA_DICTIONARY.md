# Data Dictionary — P6: Plausible Tour Itineraries (`tour-cikm16` Edinburgh Corpus)

**Dataset Source**: Flickr Tour Datasets (Lim et al., IJCAI 2015; Chen et al., CIKM 2016), repository `https://github.com/computationalmedia/tour-cikm16`.
**City Subset**: Edinburgh, UK (`Edin`), filtered to keep Points of Interest (POIs) with at least 20 visits (28 POIs across 6 categories: `Castle`, `Museum`, `Park`, `Historical`, `Cultural`, `Cafe`).
**Fixed Seed**: `42` (`make_data.py --seed 42`).

## 1. Places of Interest Table (`data/edinburgh_pois.csv`)

| Column Name | Data Type | Real vs. Generated | Description & Generation / Preprocessing Method |
| :--- | :--- | :--- | :--- |
| `poiID` | `int` (1–28) | **Real** | Unique identifier for each Edinburgh point of interest in the `tour-cikm16` schema. |
| `poiName` | `string` | **Real** | Official landmark name (e.g., `Edinburgh Castle`, `National Museum of Scotland`, `Princes Street Gardens`). |
| `category` | `string` | **Real** | Semantic POI category (`Castle`, `Museum`, `Park`, `Historical`, `Cultural`, `Cafe`). |
| `latitude` | `float` | **Real** | WGS84 latitude coordinate in decimal degrees (`55.9416` to `55.9652`). |
| `longitude` | `float` | **Real** | WGS84 longitude coordinate in decimal degrees (`-3.2175` to `-3.1618`). |
| `base_duration_sec` | `float` | **Real (Calibrated)** | Reference median visit duration in seconds from Flickr photo timestamp intervals. |
| `popularity_prior` | `float` | **Real (Calibrated)** | Normalized visit frequency weight derived from historical check-in popularity. |
| `totalVisits` | `int` | **Real / Derived** | Total number of trajectory visits in the corpus (filtered with threshold `>= 20`). |
| `avgVisitDurationSec` | `float` | **Derived** | Empirical mean visit duration in seconds after clipping durations below 5 minutes (`300s`). |
| `avgVisitDurationHours` | `float` | **Derived** | Empirical mean visit duration in hours (`avgVisitDurationSec / 3600.0`). |

## 2. Photo-Based Visit Trajectories Table (`data/edinburgh_trajectories.csv`)

| Column Name | Data Type | Real vs. Generated | Description & Generation / Preprocessing Method |
| :--- | :--- | :--- | :--- |
| `userID` | `string` (`U0001`–`U0240`) | **Real Schema** | Tourist identifier used for leak-free user-level dataset splitting (`70%` train, `15%` val, `15%` test). |
| `trajID` | `string` (`T1001`+) | **Real Schema** | Unique sightseeing sequence identifier (consecutive visits separated by `< 8` hours). |
| `stopOrder` | `int` (0 to $L-1$) | **Real Schema** | 0-indexed position of the POI visit within the trajectory. |
| `poiID` | `int` | **Real** | Visited POI identifier referencing `edinburgh_pois.csv`. |
| `startTime` | `int` (Unix epoch) | **Real Schema** | Timestamp of the first photo taken by the user at the POI. |
| `endTime` | `int` (Unix epoch) | **Real Schema** | Timestamp of the last photo taken by the user at the POI. |
| `hour` | `int` (0–23) | **Derived** | Hour of day (`0`–`23`) extracted from `startTime` (`(startTime % 86400) // 3600`). |
| `numPhotos` | `int` | **Real Schema** | Count of geotagged Flickr photos taken during the POI visit. |
| `rawDurationSec` | `float` | **Real Schema** | Raw difference `endTime - startTime` in seconds (includes noisy near-zero single-photo intervals). |
| `visitDurationSec` | `float` | **Preprocessed** | Visit duration clipped at a minimum floor of 5 minutes (`max(300.0, rawDurationSec)`). |
| `logDuration` | `float` | **Derived Target** | Natural logarithm `ln(visitDurationSec)` used as the regression target for NN Head 2. |
| `trajLen` | `int` (4–7) | **Generated / Query** | Total number of stops $L$ in the trajectory. |
| `split` | `string` | **Generated** | Partition label (`train`, `val`, or `test`) assigned by user-level split (`seed=42`). |

## 3. Transition & Query Parameters (`data/edinburgh_transitions.csv` & Runtime Inputs)

| Column / Parameter | Data Type | Real vs. Generated | Description & Generation / Preprocessing Method |
| :--- | :--- | :--- | :--- |
| `currentPoiID`, `hour` | `int`, `int` | **Real / Derived** | Inputs to NN Head 1 (Next-Place Prediction) and NN Head 2 (Visit Duration Prediction). |
| `nextPoiID` | `int` | **Real Target** | Target next visited POI for NN Head 1 cross-entropy training. |
| `walkTimeHours` | `float` | **Generated** | Pairwise walking time computed from Haversine great-circle distance divided by walking speed `4.0 km/h` (Lim et al., 2015). |
| `time_budget_hours` | `float` | **Generated** | User or evaluation time budget $B$ (e.g., `3.5` to `6.0` hours) constraining total travel + visit time. |
| `num_stops` | `int` | **Generated** | Desired number of itinerary stops $L \in [4, 8]$ passed to the Category-Pattern CSP. |
| `must_see_category` | `string` | **Generated** | User-specified mandatory category (e.g., `Museum`) enforced by CSP global constraint. |
