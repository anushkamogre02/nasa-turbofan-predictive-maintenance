# Predictive Maintenance of Turbofan Engines
## Remaining Useful Life Prediction and Health Stage Classification
### NASA CMAPSS FD001 Dataset

---

## Project Overview

Jet engines degrade over time. If an engine fails unexpectedly mid-operation,
the cost is catastrophic — emergency repairs, grounded flights, safety risks,
and regulatory consequences. The traditional approach replaces engines on a
fixed schedule regardless of actual condition — safe but wasteful.

This project builds a **data-driven predictive maintenance system** that uses
21 sensor readings collected every flight cycle to:

1. **Predict the exact number of cycles remaining** before an engine fails
   (Remaining Useful Life — RUL regression)
2. **Classify each engine into a health stage** — Healthy, Warning, or Critical
   (multi-class classification)
3. **Generate actionable maintenance decisions** with estimated cost savings

The model tells a maintenance team:
> *"Engine 47 is Critical — 18 cycles remaining. Ground it immediately."*

Rather than waiting for failure or replacing healthy engines too early.

---

## Business Impact

| Maintenance Strategy | What Happens | Estimated Cost (100 engines) |
|---|---|---|
| Reactive (fix after failure) | Engine fails mid-operation | $40M+ per cycle |
| Preventive (fixed schedule) | Replace on time regardless of condition | $12M per cycle |
| **Predictive (this model)** | **Act only when data says so** | **$9–11M per cycle** |

**Estimated savings: ~$29 million per 100-engine fleet per maintenance cycle**
by shifting unplanned failures ($2M each) to planned maintenance ($150K each).

> Note: Cost figures are illustrative approximations based on industry
> estimates. Actual savings depend on fleet size and maintenance contracts.

---

## Dataset

**Source:** NASA Prognostics Center of Excellence — CMAPSS
(Commercial Modular Aero-Propulsion System Simulation)

**Download:**
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

| File | Description |
|---|---|
| `train_FD001.txt` | 100 engines run until failure — full life history |
| `test_FD001.txt` | 100 engines stopped before failure — predict remaining life |
| `RUL_FD001.txt` | True RUL answer sheet for test engines |

**Dataset structure:**
- 20,631 training rows (100 engines × average ~200 cycles each)
- 26 columns per row: engine ID + cycle + 3 operational settings + 21 sensors
- One fault mode (FD001 is the simplest and most studied CMAPSS subset)

---

## Key Results

### Regression — Predicting Exact RUL

| Model | RMSE (cycles) | R² | Notes |
|---|---|---|---|
| Linear Regression | 37.54 | 0.184 | Baseline |
| Decision Tree | 25.37 | 0.627 | Controlled depth |
| **Random Forest** | **22.51** | **0.707** | **Best model** |
| XGBoost | ~23.0 | ~0.690 | Comparable to RF |

**5-Fold Cross Validation — Random Forest:**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|---|---|---|---|---|---|---|
| 17.76 | 20.45 | 20.14 | 20.99 | 20.84 | **20.04** | **1.18** |

Low std (1.18) confirms results are consistent and reliable — not a lucky split.

### Classification — Health Stage Assignment (100 Test Engines)

| Stage | RUL Threshold | Engines Flagged | Action |
|---|---|---|---|
| 🟢 Healthy | RUL > 100 cycles | 33 engines | Routine monitoring |
| 🟡 Warning | RUL 30–100 cycles | 42 engines | Schedule maintenance this week |
| 🔴 Critical | RUL < 30 cycles | 25 engines | Ground for immediate inspection |

### Feature Importance — Top Finding

All **top 10 most important features** were rolling mean features —
not raw sensor values. The single most important feature was
`s_4_rmean` at approximately **42% importance**.

This validates the core engineering decision:
*degradation trend matters more than a single sensor snapshot.*

---

## Methodology

### Step 1 — Data Understanding and EDA

- Loaded 3 NASA text files with custom column naming (no headers in source)
- Calculated RUL for each training row:
  `RUL = max_cycle_of_engine − current_cycle`
- Removed 10 constant columns (zero standard deviation = zero information):
  `setting_1, setting_2, setting_3, s_1, s_5, s_6, s_10, s_16, s_18, s_19`
- Analysed engine lifespan: median ~160 cycles, range 31–361 cycles
- Plotted all 14 remaining sensors vs RUL — confirmed sensors only respond
  meaningfully when RUL drops below 125 cycles

### Step 2 — Feature Engineering (The Key Upgrade)

**Problem with raw sensor values:**
A single reading at one moment is a snapshot.
Engine degradation is a process that happens over time — the trend matters.

**Solution — rolling window features (window = 15 cycles):**

For each of the 12 selected sensors:
- `sensor_rmean` = rolling mean → captures trend direction
- `sensor_rstd`  = rolling std  → captures instability near failure

Grouped by engine before calculating to ensure no data leakage between engines.

**Why this matters:**
Feature importance showed all top 10 features were rolling mean features.
`s_4_rmean` alone accounted for 42% of total feature importance.
Raw sensor values barely appeared in the top 15.

### Step 3 — RUL Clipping at 125

Sensors are flat when RUL > 125 — the model cannot distinguish an engine
with 300 cycles remaining from one with 200 cycles remaining.
Clipping at 125 focuses training on the window where sensor data actually
carries degradation information.

### Step 4 — Feature Selection

Selected 12 sensors with absolute correlation ≥ 0.5 with RUL:

```
s_2, s_3, s_4, s_7, s_8, s_9, s_11, s_12, s_13, s_14, s_15, s_17
```

Removed weak sensors reduce noise and improve model generalisation.

### Step 5 — Scaling

StandardScaler applied to all features.
Fitted on training data only — applied to test without refitting.
This prevents data leakage (a common mistake in beginner projects).

### Step 6 — Regression Models

Trained and compared 4 models with 3 evaluation metrics each:

**Models:**
- Linear Regression — baseline, simplest possible model
- Decision Tree — explainable, max_depth=8 to prevent overfitting
- Random Forest — bagging ensemble, 100 trees, best performer
- XGBoost — gradient boosting, sequential error correction

**Evaluation metrics:**
- **RMSE** — average prediction error in cycles (lower = better)
- **R²** — fraction of RUL variation explained (higher = better)
- **NASA Asymmetric Score** — from original CMAPSS research papers,
  penalises late predictions more than early ones (lower = better)

### Step 7 — Health Stage Classification

Converted numeric RUL to 3 actionable categories:

```
Healthy  (class 0): RUL > 100  →  routine monitoring
Warning  (class 1): RUL 30–100 →  schedule maintenance this week
Critical (class 2): RUL < 30   →  ground for immediate inspection
```

Trained 3 classifiers: Logistic Regression, Random Forest, XGBoost.
Evaluated with weighted F1 score (correct metric for imbalanced classes).

Confusion matrix analysis identified the most dangerous mistake:
**Critical classified as Healthy** = engine may fail before maintenance.
Model tuning prioritised recall for the Critical class specifically.

### Step 8 — Business Output

- Daily fleet health dashboard — clear action per engine
- Cost savings analysis connecting model output to financial impact

---

## Project Structure

```
nasa-turbofan-predictive-maintenance/
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── NASA_Turbofan_Predictive_Maintenance-FD001.ipynb     
└── README_Nasa_Turbofan.md                 
```

---

## How to Run

**Step 1 — Clone the repository**

```bash
git clone https://github.com/anushkamogre02/nasa-turbofan-predictive-maintenance.git
cd nasa-turbofan-predictive-maintenance
```

**Step 2 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3 — Download the dataset**

Download from NASA:
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Place these 3 files inside the `data/` folder:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

****Step 4 — Launch Jupyter Notebook**

```bash
jupyter notebook
```

Open:

NASA_Turbofan_Predictive_Maintenance-FD001.ipynb
```

and run all cells sequentially.

---

### Option B — Run using Google Colab

1. Upload the notebook:

```
NASA_Turbofan_Predictive_Maintenance-FD001.ipynb
```

to Google Colab.

2. Upload the dataset files into the Colab session.

3. Run all notebook cells.

---

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
jupyter>=1.0.0 

---

*Built with Python · scikit-learn · XGBoost · pandas · matplotlib · seaborn*

*Dataset: NASA CMAPSS FD001 — Prognostics Center of Excellence*
