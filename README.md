# Public Safety ML — Toronto KSI Collision Analysis

A supervised machine learning project analyzing Toronto's **Killed or Seriously Injured (KSI)** traffic collision dataset to identify patterns and predict collision severity.

## Project Overview

This project performs end-to-end data preprocessing and exploratory data analysis on the Toronto KSI dataset, preparing it for machine learning model training. The goal is to understand the factors contributing to fatal and serious-injury collisions in Toronto.

## Dataset

**KSI_data.csv** — Toronto Police Service KSI collision records.

Key features include:
- Collision details: `ACCLASS`, `ACCLOC`, `IMPACTYPE`, `ROAD_CLASS`, `TRAFFCTL`
- Environmental conditions: `LIGHT`, `VISIBILITY`, `RDSFCOND`
- Involvement details: `INVTYPE`, `INJURY`, `PEDESTRIAN`, `CYCLIST`, `AUTOMOBILE`
- Driver behaviour: `SPEEDING`, `AG_DRIV`, `REDLIGHT`, `ALCOHOL`, `DISABILITY`
- Location: `DISTRICT`, `STREET2`, `INITDIR`, `OFFSET`

## Project Structure

```
GroupProject/
├── KSI_data.csv          # Toronto KSI collision dataset
├── public_safety_ml.py   # Data loading, EDA, and preprocessing pipeline
└── README.md             # Project documentation
```

## Preprocessing Pipeline

The missing-value strategy is domain-driven, treating each group of columns according to its real-world semantics:

| Group | Columns | Strategy | Rationale |
|-------|---------|----------|-----------|
| **1 — Binary flags** | `PEDESTRIAN`, `CYCLIST`, `ALCOHOL`, `SPEEDING`, etc. | `fillna("No")` | NaN = not involved |
| **2 — Role-conditional** | `PEDTYPE`, `PEDACT`, `CYCLISTYPE`, `CYCACT`, etc. | `fillna("Not Applicable")` | Only applies to pedestrians/cyclists |
| **3A — True missing (<3%)** | `ACCLASS`, `LIGHT`, `VISIBILITY`, `TRAFFCTL`, etc. | Drop rows | Genuinely incomplete reports |
| **3B — True missing (~28%)** | `ACCLOC`, `INITDIR` | `fillna("Unknown")` | Too many to drop |
| **4 — Identifiers** | `OFFSET`, `STREET2`, `VEHTYPE`, `ACCNUM` | Fill or placeholder | Not analysis variables |
| **5 — Driver-specific** | `MANOEUVER`, `DRIVACT`, `DRIVCOND` | `fillna("Not Applicable")` | Role-driven missingness |
| **6 — Injury** | `INJURY` | Conditional fill by `INVTYPE` | "None" for scene-present roles; "Not Applicable" for bystanders |

## Requirements

```
pandas
scikit-learn
```

Install dependencies:
```bash
pip install pandas scikit-learn
```

## Usage

```bash
python public_safety_ml.py
```

## Course

**COMP247 — Supervised Learning**  
Centennial College — Semester 4
