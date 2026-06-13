# Public Transport & Logistics Demand Forecasting Pipeline

An end-to-end, high-performance machine learning pipeline designed to predict spatial-temporal demand using CatBoost Regression. The architecture leverages Out-Of-Fold (OOF) Target Encoding for high-cardinality spatial identifiers and extracts nuanced temporal characteristics to maximize predictive accuracy ($R^2$).

---

## 🚀 System Architecture & Workflow

The pipeline processes raw spatial-temporal data points through structured feature engineering, robust target encoding to prevent data leakage, and training validation loops.



---

##  Features Engineered

### 1. Temporal Deconstruction
* **Linear Decomposition:** Extracts `hour`, `minute`, and absolute `time_in_mins` from raw string timestamps.
* **Domain Flags:** Implements binary features capturing situational demand spikes:
  * `rush_hour`: Captures peak daily transit hours `[7-9 AM, 17-19 PM]`.
  * `night`: Flags low-frequency nocturnal intervals `[22 PM - 5 AM]`.
* **Cyclical Time Encoding (Planned):** Mapping linear time to 2D periodic coordinates using Sine and Cosine transformations to represent seamless transitions from midnight to early morning.

### 2. Spatial Mapping & High-Cardinality Encoding
* **Out-of-Fold (OOF) Target Encoding:** Computes cross-validated target statistics on geographic `geohash` inputs, protecting tree splits from severe over-fitting and data leakage.
* **Spatial Prefixing (Planned):** Slicing categorical geohash IDs hierarchically (`str[:3]`, `str[:4]`) to dynamically map macro-regional hubs.

---

##  Experiment Validation Ledger

A systematic log tracking architectural modifications, model iterations, and validation benchmarks ($R^2$ Score evaluation).

| Pipeline Version | Core Model Configuration | Engineering Strategy Implemented | Validation $R^2$ Score | Execution Status |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0 (Baseline)** | `CatBoostRegressor` | Base Time Parsing + OOF Geohash Target Encoding | **0.9464** | `Merged to Main` |
| **v1.1 (Target)** | `CatBoostRegressor` | Cyclical Time (Sin/Cos) + Geohash Sub-string Grids | *TBD* | `In Progress` |
| **v1.2 (Planned)** | `Ensemble (XGB/LGBM)`| Weighted Blending of Gradient Boosted Architectures | *TBD* | `Pipeline Design` |

---

##  Tech Stack & Dependencies

* **Language:** Python
* **Modeling & Optimization:** `catboost`, `scikit-learn`
* **Data Manipulation:** `pandas`, `numpy`
* **Categorical Handling:** `category_encoders`

---

##  Project Directory Configuration

```text
├── data/
│   ├── train.csv                      # Raw training observations (Git Ignored)
│   └── test.csv                       # Unseen testing matrix (Git Ignored)
├── .gitignore                         # Strategic tracking exclusion boundaries
├── Untitled4.ipynb                    # End-to-end training notebook pipeline
├── submission.csv                     # Compiled target outputs (Git Ignored)
└── README.md                          # Production documentation
