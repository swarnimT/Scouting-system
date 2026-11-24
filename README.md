<div align="center">

<img src="assets/logo.png" alt="SCOUTIX Logo" width="340"/>

# SCOUTIX — European Football Scout AI

**Next-Generation AI & Unsupervised Machine Learning Scouting System across Europe's Big 5 Leagues**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests Passing](https://img.shields.io/badge/pytest-36%20passed-success?logo=pytest&logoColor=white)](tests/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Analyse • Scout • Discover*

</div>

---

## 📌 Overview

**SCOUTIX** is an advanced AI-driven football scouting platform designed to identify tactically similar players across Europe's top 5 leagues (**Premier League, La Liga, Serie A, Bundesliga, and Ligue 1**). 

By uniting comprehensive per-90 rate metrics from **FBref (2024–2025 season)** with player valuations, photos, and physical profiles from **Transfermarkt**, SCOUTIX enables sporting directors, scouts, and data analysts to:
- Find tactical replacements for outgoing stars or aging squad members.
- Discover undervalued U-23 talent across alternative European leagues.
- Filter players dynamically by age cap, league combinations, position lock, and club exclusion.
- Compare profiles through multi-metric head-to-head heatmaps, polar radar charts, and 2D tactical archetype maps.

---

## 🏗️ System Architecture

```
 Raw FBref Stats (2024–2025)           Transfermarkt Players Database
  [players_data-2024_2025.csv]                  [players.csv]
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                         src/preprocess.py
               • Clean country prefixes ('eng ', 'es ')
               • Consolidate 152 mid-season transfers
               • Filter outfield players (Min. 10.0 90s)
               • Two-tier Transfermarkt enrichment (bio + € valuation)
               • Compute 14 per-90 metrics & position percentiles
                                  │
                                  ▼
                         src/clustering.py
               • StandardScaler normalization
               • KMeans clustering (k=6 tactical archetypes)
               • 2D PCA projection (66.6% explained variance)
                                  │
                                  ▼
              data/processed/players_processed.parquet
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       ▼                                                     ▼
src/similarity.py                                     app/visuals.py
 • Pairwise Cosine Similarity matrix                   • Plotly Polar Radar Chart
 • 0–100% normalized match score                       • 2D PCA Tactical Archetype Map
 • Multi-constraint filtering engine                   • Highlighting target & matches
       │                                                     │
       └──────────────────────────┬──────────────────────────┘
                                  ▼
                             app/main.py
               • SCOUTIX Streamlit multi-panel dashboard
               • Live search dropdown with autocomplete
               • Target player bio card (photo, € valuation, foot, height)
               • 3-column tactical match scorecards
               • Head-to-Head metric table with gradient styling
               • Interactive Radar and Cluster tabs
```

---

## 📂 Repository Layout

```
scouting system/
├── assets/
│   └── logo.png                        # SCOUTIX brand logo
├── data/
│   ├── raw/                            # Raw data inputs (git-ignored)
│   │   ├── players_data-2024_2025.csv  # 2024–2025 FBref Big 5 dataset
│   │   └── players.csv                 # Transfermarkt global database
│   ├── processed/                      # Intermediate & final datasets (git-ignored)
│   │   ├── players_preprocessed.parquet
│   │   ├── players_processed.parquet
│   │   └── X_scaled.npy
│   ├── players_processed.parquet       # Engine-ready dataset (1,460 players)
│   └── README_data.md                  # Data contract specification
├── models/
│   └── scaler.joblib                   # Fitted StandardScaler model
├── src/
│   ├── __init__.py
│   ├── preprocess.py                   # Feature engineering & Transfermarkt merge
│   ├── clustering.py                   # KMeans clustering & PCA projection
│   └── similarity.py                   # Pairwise Cosine Similarity engine
├── app/
│   ├── __init__.py
│   ├── main.py                         # Streamlit multi-panel dashboard
│   └── visuals.py                      # Plotly radar & cluster map visualizations
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Synthetic data fixtures for testing
│   ├── test_similarity.py              # 30 unit tests for similarity engine
│   ├── test_visuals.py                 # 3 unit tests for Plotly visualizations
│   └── test_real_data.py               # 3 integration tests on real European data
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated GitHub Actions test pipeline
├── .gitignore                          # Clean repository rules
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.11+** (tested and fully compatible with Python 3.11, 3.12, and 3.14)
- **PowerShell** or any standard terminal

### 2. Installation

Clone the repository and install dependencies:

```bash
# Clone repository
git clone https://github.com/<YOUR_ORG>/scouting-system.git
cd "scouting system"

# Install required dependencies
py -m pip install -r requirements.txt
```

### 3. Generate or Refresh Processed Data

To run the complete data engineering and ML pipeline from raw files:

```bash
# Step 1: Clean stats, resolve mid-season transfers, and merge Transfermarkt bio
py -m src.preprocess

# Step 2: Fit StandardScaler, evaluate KMeans clusters, and fit 2D PCA
py -m src.clustering
```

### 4. Launch the SCOUTIX Dashboard

Launch the interactive web application:

```bash
py -m streamlit run app/main.py
```

Your browser will automatically open:
👉 **`http://localhost:8501`**

---

## 🧠 Machine Learning & Tactical Methodology

### 1. The 14 Core Per-90 Rate Metrics
To capture playing style rather than playing volume, all metrics are normalized per 90 minutes (`stat / 90s`):

| Metric | Source Stat | Tactical Dimension |
| :--- | :--- | :--- |
| **`goals_per90`** | `Gls` | Finishing / Goal Threat |
| **`assists_per90`** | `Ast` | Chance Creation / Final Pass |
| **`xg_per90`** | `npxG` | Non-Penalty Expected Goals Quality |
| **`xa_per90`** | `xAG` | Expected Assisted Goals Quality |
| **`shots_per90`** | `Sh` | Shot Volume & Directness |
| **`key_passes_per90`** | `SCA` / `KP` | Shot-Creating Actions / Key Passes |
| **`progressive_carries_per90`** | `PrgC` | Ball Progression via Dribbling |
| **`progressive_passes_per90`** | `PrgP` | Ball Progression via Passing |
| **`dribbles_per90`** | `Succ` | Take-On Frequency & 1v1 Skill |
| **`pressures_per90`** | `Blocks` | Defensive Disruptions & Blocking |
| **`tackles_per90`** | `TklW` | Tackles Won / Ground Duels |
| **`interceptions_per90`** | `Int` | Defensive Anticipation & Interceptions |
| **`aerials_won_per90`** | `Won` | Aerial Duel Dominance |
| **`touches_att_pen_per90`** | `Att Pen` | Penalty Box Occupation |

### 2. Vector Similarity Engine
- **Algorithm:** Pairwise **Cosine Similarity** over the 14 standardized features.
- **Why Cosine Similarity?** Cosine similarity evaluates the angle between high-dimensional style vectors rather than magnitude, ensuring that stylistic shape is matched independently of raw team possession dominance.
- **Match Index:** Normalized from $[-1, 1]$ into an intuitive **0–100% Match Percentage**:
  $$\text{Match \%} = \frac{\cos(\mathbf{u}, \mathbf{v}) + 1}{2} \times 100$$

### 3. Tactical Archetypes ($k=6$ KMeans)
 Centroid analysis maps players into 6 distinct tactical clusters:
- **Cluster 0 — Central Defender / Interceptor:** High defensive interceptions, low progressive passes.
- **Cluster 1 — Attacking Fullback / Crosser:** High crossing volume (`Crs_per90` $z=+1.44$), wide shot creation.
- **Cluster 2 — Playmaker / Inverted Winger:** Exceptional expected assisted goals (`xAG` $z=+1.86$), shot creation, and progressive carries.
- **Cluster 3 — Ball-Winning Midfielder:** High ground tackles won (`TklW` $z=+0.89$), interceptions, and central ball progression.
- **Cluster 4 — Target Forward / Finisher:** Peak non-penalty expected goals (`npxG` $z=+1.86$), goal conversion, and box presence.
- **Cluster 5 — Direct Winger / Ball Carrier:** High take-on dribbles (`Succ` $z=+0.78$), carries, and transition speed.

---

## 🧪 Testing & Verification

The project includes a comprehensive automated test suite with **36 automated tests** (100% pass rate):

```bash
# Run all tests
py -m pytest tests/ -v

# Run tests with detailed code coverage report
py -m pytest tests/ -v --cov=src --cov=app --cov-report=term-missing
```

### Coverage Overview

| Test Module | Coverage Scope | Tests | Status |
| :--- | :--- | :---: | :---: |
| **`test_similarity.py`** | Matrix symmetry, range $[0, 100]$, self-exclusion, all filters, schema validation | 30 | ✅ PASSED |
| **`test_visuals.py`** | Polar radar chart generation & 2D PCA cluster map rendering | 3 | ✅ PASSED |
| **`test_real_data.py`** | Integration with live 2024–2025 European dataset and Transfermarkt valuations | 3 | ✅ PASSED |
| **Total** | Full end-to-end pipeline verification | **36** | **✅ 100% PASS** |

---

## 🛠️ Tech Stack

- **Core Data & ML:** Python, `pandas`, `numpy`, `scikit-learn`, `pyarrow`
- **Frontend Dashboard:** `Streamlit`
- **Visualizations:** `Plotly` (Graph Objects & Express), `matplotlib`
- **Testing & Quality:** `pytest`, `pytest-cov`, GitHub Actions CI
- **Data Providers:** FBref (Stats), Transfermarkt (Valuations & Bio)

---

## 👥 Contributors

- **SCOUTIX Engineering Team**
  - **Partner 1:** Feature Engineering, Transfermarkt Data Enrichment, KMeans Clustering, and 2D PCA Projections.
  - **Partner 2:** Vector Similarity Engine, Streamlit UI Architecture, Caching, and Test Automation.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to modify and expand for professional or academic scouting.
