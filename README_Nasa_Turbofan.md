# Predictive Maintenance of Turbofan Engines Using Machine Learning

**NASA CMAPSS FD001 Dataset — RUL Prediction and Health Stage Classification**

---

## Overview

Jet engines degrade over time and eventually fail. Unexpected failure costs
approximately **$2 million per incident** in emergency repairs, grounded flights,
and regulatory consequences. Replacing engines on a fixed schedule wastes money
on healthy engines that still have significant life remaining.

This project builds a **data-driven predictive maintenance system** using
21 sensor readings collected every flight cycle to:

- Predict the exact number of cycles remaining before an engine fails (**RUL Regression**)
- Classify each engine as **Healthy**, **Warning**, or **Critical** (**Health Stage Classification**)
- Generate a **daily fleet health dashboard** with maintenance actions and cost savings

---

## Business Impact

| Strategy | Description | Estimated Cost (100 engines) |
|---|---|---|
| Reactive | Fix after failure | $40M+ per cycle |
| Preventive | Replace on fixed schedule | $12M per cycle |
| **Predictive (this project)** | **Act only when data says so** | **~$9–11M per cycle** |

**Estimated savings: ~$29 million per 100-engine fleet per maintenance cycle**

> Costs are illustrative approximations based on industry estimates.

---

## Dataset

**Source:** NASA Prognostics Center of Excellence — CMAPSS
(Commercial Modular Aero-Propulsion System Simulation)

**Download:** https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

| File | Description |
|---|---|
| `train_FD001.txt` | 100 engines run until failure — full life history |
| `test_FD001.txt` | 100 engines stopped before failure — predict remaining life |
| `RUL_FD001.txt` | True RUL answer sheet for test engines |

- **Training rows:** 20,631 (100 engines × ~200 cycles average)
- **Columns:** 26 — engine ID, cycle, 3 operational settings, 21 sensor readings
- **Fault modes:** 1 (FD001 is the most studied CMAPSS subset)

---

## Results

### Regression — Predicting Exact RUL

| Model | RMSE (cycles) | R² |
|---|---|---|
| Linear Regression | 37.54 | 0.184 |
| Decision Tree | 25.37 | 0.627 |
| **Random Forest** | **22.51** | **0.707** |
| XGBoost | ~23.0 | ~0.690 |

**5-Fold Cross Validation — Random Forest**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|---|---|---|---|---|---|---|
| 17.76 | 20.45 | 20.14 | 20.99 | 20.84 | **20.04** | **1.18** |

Low std (1.18) confirms model is consistent and reliable across all splits.

### Classification — Health Stage (100 Test Engines)

| Stage | RUL Threshold | Engines | Action |
|---|---|---|---|
| 🟢 Healthy | > 100 cycles | 33 | Routine monitoring |
| 🟡 Warning | 30–100 cycles | 42 | Schedule maintenance this week |
| 🔴 Critical | < 30 cycles | 25 | Ground for immediate inspection |

### Feature Importance

All **top 10 features** were rolling mean features — not raw sensor values.
Top feature: **s_4_rmean at ~42% importance**.

This confirms the core engineering decision: **degradation trend beats snapshot.**

---

## Methodology

### 1. Data Preparation
- Calculated RUL: `RUL = max_cycle_of_engine − current_cycle`
- Removed 10 constant columns (zero variance = zero information)
- Selected 12 sensors with absolute correlation ≥ 0.5 with RUL

### 2. Feature Engineering
Raw sensor readings are snapshots. Engine degradation is a trend over time.

For each of the 12 selected sensors, over a **15-cycle rolling window**:
- `sensor_rmean` — rolling mean → captures trend direction
- `sensor_rstd`  — rolling std  → captures instability near failure

Calculated **per engine** using `groupby` to prevent data leakage between engines.

### 3. RUL Clipping at 125
Sensors are flat when RUL > 125. The model cannot distinguish 300 cycles
remaining from 200. Clipping focuses training on the window where sensor
data carries actual degradation information.

### 4. Scaling
`StandardScaler` fitted on training data only — applied to test without
refitting. Prevents data leakage.

### 5. Regression Models
Four models trained and compared using three metrics:
- **RMSE** — average prediction error in cycles
- **R²** — fraction of RUL variation explained
- **NASA Asymmetric Score** — penalises late predictions more than early ones
  (from the original CMAPSS research papers)

### 6. Health Stage Classification
Three classifiers trained: Logistic Regression, Random Forest, XGBoost.
Evaluated with **weighted F1 score** (correct metric for imbalanced classes).

Confusion matrix analysis identifies the most dangerous mistake:
**Critical classified as Healthy** — engine may fail before maintenance arrives.

### 7. Business Output
- Daily fleet health dashboard with action per engine
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
├── NASA_Turbofan_Predictive_Maintenance_FD001.ipynb
├── requirements.txt
└── README_nasa_turbofan.md
```

---

## How to Run

### Option A — Google Colab (recommended, no setup needed)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `NASA_Turbofan_Predictive_Maintenance_FD001.ipynb`
3. Upload the 3 data files to the Colab session storage
4. Run all cells top to bottom

### Option B — Jupyter Notebook (local)

```bash
# Clone the repository
git clone https://github.com/anushkamogre02/nasa-turbofan-predictive-maintenance.git
cd nasa-turbofan-predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Open `NASA_Turbofan_Predictive_Maintenance_FD001.ipynb` and run all cells.

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
```

---

## Key Differentiators

**1. Rolling Window Features**
Raw sensor values are snapshots. This project engineers rolling mean and
standard deviation over 15-cycle windows — capturing degradation trends
rather than isolated readings. Feature importance validated this approach:
all top 10 features were rolling mean features.

**2. NASA Asymmetric Evaluation**
Uses NASA's official scoring function from the original CMAPSS research papers —
not just RMSE. Late predictions are penalised more than early ones because
predicting an engine has more life than it does is operationally dangerous.

**3. Classification Layer**
Health stage classification converts numeric RUL into actionable decisions.
Maintenance teams act on *"CRITICAL"*, not on *"47.3 cycles remaining"*.

---

*Built with Python · scikit-learn · XGBoost · pandas · matplotlib · seaborn*
*Dataset: NASA CMAPSS FD001 — NASA Prognostics Center of Excellence*
