# Public Safety ML — Toronto KSI Collision Analysis

A supervised machine learning project analyzing Toronto's **Killed or Seriously Injured (KSI)** traffic collision dataset to understand what drives fatal outcomes on Toronto's roads, and to predict collision severity from the conditions present at the scene.

---

## 1. The Problem

Every year, thousands of traffic collisions occur across the City of Toronto. Most result in property damage or minor injury — but a subset end in death or serious, life-altering injury. These are the collisions captured in the **KSI (Killed or Seriously Injured)** dataset published by the Toronto Police Service.

The core problem this project addresses:

> **Given the conditions of a collision — location, time, road/weather conditions, who and what was involved, and driver behaviour — can we predict whether it will be fatal, and which factors matter most?**

This is fundamentally a **road-safety and public-policy problem**, not just a data problem. Traffic fatalities are largely preventable — they result from a combination of infrastructure design, enforcement gaps, environmental conditions, and human behaviour. Understanding *which combinations of factors* push a collision from "serious" to "fatal" is the first step toward preventing it.

## 2. Why This Matters — The Business Perspective

Toronto operates under **Vision Zero**, the city's road-safety plan aiming to eliminate traffic fatalities and serious injuries. A model like this one supports that goal by turning historical collision records into **actionable risk intelligence**. Several stakeholders have a direct interest in this analysis:

| Stakeholder | Business Need | How This Project Helps |
|---|---|---|
| **Toronto Police Service / City of Toronto (Vision Zero)** | Prioritize enforcement and infrastructure spending where it saves the most lives | Identifies the strongest predictors of fatality (e.g. speeding, pedestrian involvement, lighting/road conditions) to target interventions |
| **Urban planners / Transportation Services** | Decide where to invest in traffic calming, better lighting, redesigned intersections | Neighbourhood- and location-level patterns highlight high-risk areas and conditions |
| **Insurance providers** | Better understand risk factors that correlate with severe outcomes | Feature importance from the model reflects real-world severity drivers, not just claim frequency |
| **Emergency services** | Anticipate when/where KSI-severity collisions are more likely | Time-of-day and seasonal patterns inform resource and ambulance staffing |
| **Public awareness campaigns** | Target messaging (e.g. anti-speeding, anti-impaired-driving campaigns) | Quantifies how much specific behaviours (speeding, alcohol, red-light running) contribute to fatal outcomes |

The economic and human cost of a fatal collision is enormous — beyond the immediate loss of life, there are long-term healthcare costs, legal and insurance costs, lost productivity, and lasting impact on families and communities. Even modest improvements in identifying and mitigating high-risk conditions translate into real lives saved and costs avoided.

## 3. Project Goal

Frame this as a **binary classification problem**:

- **Target variable:** `ACCLASS_BINARY` — `1 = Fatal`, `0 = Non-Fatal` (property-damage and non-fatal-injury outcomes are combined, since the business question is "fatal vs. not")
- **Objective:** Train a model that predicts the likelihood a given collision will be fatal, based on features known at (or shortly after) the time of the incident, while avoiding any feature that would leak the outcome itself (e.g. injury severity fields).

## 4. Dataset

**`KSI_data.csv`** — Toronto Police Service KSI collision records, spanning 2006–present.

Key feature groups:
- **Collision details:** `ACCLASS`, `ACCLOC`, `IMPACTYPE`, `ROAD_CLASS`, `TRAFFCTL`
- **Environmental conditions:** `LIGHT`, `VISIBILITY`, `RDSFCOND`
- **Involvement details:** `INVTYPE`, `INJURY`, `PEDESTRIAN`, `CYCLIST`, `AUTOMOBILE`, `MOTORCYCLE`, `TRUCK`
- **Driver behaviour flags:** `SPEEDING`, `AG_DRIV` (aggressive driving), `REDLIGHT`, `ALCOHOL`, `DISABILITY`
- **Location:** `DISTRICT`, `HOOD_158`/`HOOD_140` (neighbourhood codes), `LATITUDE`, `LONGITUDE`

A significant part of the work in this project is *not* the modelling itself, but the disciplined handling of missingness — in this dataset, "missing" rarely means "unknown." It usually means "not applicable" (e.g. a pedestrian-only field left blank for a driver) or "did not occur" (e.g. `ALCOHOL` blank means alcohol was not a factor). Treating every blank field as generically missing would destroy the signal in the data, so the preprocessing pipeline applies a different, justified strategy to each semantic group of columns rather than one blanket approach.

## 5. Project Structure

```
GroupProject/
├── KSI_data.csv                       # Toronto KSI collision dataset
├── public_safety_ml.py                # Exploration -> Visualization -> Modelling pipeline
├── 1_collisions_by_year_severity.png  # Yearly trend, by severity
├── 2_collisions_by_hour.png           # Hour-of-day risk profile
├── 3_contributing_factors.png         # Frequency of behavioural risk factors
├── 4_correlation_heatmap.png          # Relationships between contributing factors
├── 5_top_neighbourhoods.png           # Highest-collision neighbourhoods
└── README.md                          # Project documentation
```

## 6. Methodology

The pipeline runs in three deliberate stages, in this order:

### Stage 1 — Data Exploration & Cleaning
- Profiles the dataset (shape, types, missingness, ranges) to understand what we're working with before touching it.
- Applies a **domain-driven missing-value strategy**, split into six groups by *why* the data is missing rather than treating every NaN the same way:

| Group | Example Columns | Strategy | Business Rationale |
|-------|---------|----------|-----------|
| **1 — Binary involvement flags** | `PEDESTRIAN`, `ALCOHOL`, `SPEEDING` | `fillna("No")` | Missing = factor was not present, not unknown |
| **2 — Role-conditional attributes** | `PEDTYPE`, `PEDACT`, `CYCLISTYPE` | `fillna("Not Applicable")` | Only relevant to pedestrians/cyclists — doesn't apply to a driver-only record |
| **3 — Event-conditional (`FATAL_NO`)** | `FATAL_NO` | `fillna(0)` | Only populated for fatal collisions |
| **4 — Genuinely missing reports** | `ACCLOC`, `INITDIR`, `ROAD_CLASS` | Drop (<3%) or `"Unknown"` (~28%) | Real reporting gaps, handled by size of the gap |
| **5 — Identifiers / low-value fields** | `OFFSET`, `STREET2`, `ACCNUM` | Fill or drop | Location/ID descriptors, not analytical signal |
| **6 — Role-dependent `INJURY`** | `INJURY` | Conditional fill by `INVTYPE` | "None" for people present at the scene; "Not Applicable" for bystander roles |

### Stage 2 — Data Visualization
Once the data is clean, exploratory charts surface the patterns a business stakeholder would actually ask about:

- **Collisions by year and severity** — Total KSI volume trends downward over the period, from ~1,400+ collisions/year in 2006–2007 to roughly 600–650/year by 2020–2022 (with a step-down starting in 2020, likely reflecting reduced traffic during COVID). However, the **fatal (red) band barely shrinks** even as total volume roughly halves — fatal counts hover in the 100–200/year range across almost the entire period. In other words: fewer collisions overall, but a similar number are still deadly, meaning the *share* of KSI collisions that are fatal has been creeping upward. This is a critical finding for Vision Zero — raw collision reduction alone isn't translating into a proportional fatality reduction.
- **Collisions by hour of day** — Non-fatal collision volume climbs steadily from a 3–4am low to a clear evening-commute peak around 5–7pm, tracking overall traffic volume. Fatal collisions don't follow that same curve — they stay comparatively substantial even during low-traffic overnight hours. That's a classic road-safety pattern: fewer crashes at night, but a higher proportion of them are severe, consistent with higher speeds, lower visibility, and more impaired driving overnight. This has direct implications for enforcement and emergency-resource scheduling.
- **Contributing factors** — Aggressive driving (`AG_DRIV`) dominates, appearing in roughly 4x as many KSI collisions as the next-most-common factor, speeding (~2,400), followed by red-light running (~1,600), then alcohol and disability much further behind (under 1,000 each). This ranks where enforcement and public-awareness spending would have the most reach — aggressive driving as a category outweighs the other behavioural factors combined.
- **Correlation heatmap** — Most risk factors are only weakly related to each other, which is itself informative: these are largely *independent* risk pathways rather than one "bad driver" profile causing everything. The exceptions worth noting: speeding and aggressive driving (0.39 correlation), and aggressive driving and red-light running (0.29), cluster together — suggesting a genuine "reckless driving" pattern rather than three unrelated issues, and a case for combined enforcement targeting all three at once. Pedestrian involvement is mildly negatively correlated with cyclist and motorcycle involvement (-0.28, -0.23), indicating these tend to be distinct collision types rather than co-occurring.
- **Top neighbourhoods** — Waterfront Communities-The Island leads by a wide margin (~730 collisions), ahead of West Humber-Clairville (~570) and Bay Street Corridor (~440). The list mixes two distinct neighbourhood profiles: dense downtown cores with heavy pedestrian exposure (Waterfront Communities, Bay Street Corridor, Church-Yonge Corridor) and outer/suburban areas with higher-speed arterials (West Humber-Clairville, Woburn, Rouge, Wexford/Maryvale). This matters for policy — a downtown intersection redesign and a suburban speed-enforcement push are different interventions, but both neighbourhood types show up as high-KSI zones and likely need tailored responses.

### Stage 3 — Data Modelling
Builds directly on the cleaned exploration output (no re-loading or re-imputing):

1. **Target preparation** — collapses the 3-class outcome into a binary `Fatal` vs `Non-Fatal` target.
2. **Leakage removal** — drops any column that describes the outcome itself (`INJURY`, `FATAL_NO`), since including these would let the model "cheat" rather than predict from causal/contextual factors.
3. **Feature selection with justification** — every dropped and kept column is documented with a business/statistical reason (sparsity, redundancy, or genuine predictive relevance).
4. **Feature engineering** — extracts `HOUR` from raw `TIME`, groups rare categories (<1%) into `"Other"` to avoid noisy one-hot columns, and encodes binary flags as 0/1.
5. **Stratified 80/20 train/test split** — preserves the real-world fatal/non-fatal ratio in both sets so evaluation reflects reality.
6. **Preprocessing pipeline** — numeric features are median-imputed and standardized; categorical features are mode-imputed and one-hot encoded, all inside a single `ColumnTransformer` fit only on training data (no test-set leakage).
7. **Class imbalance handling (SMOTE)** — fatal collisions are a small minority of all KSI records. SMOTE is applied **only to the training set** to avoid inflating evaluation metrics with synthetic test data, since a model that just predicts "non-fatal" every time would score misleadingly well on accuracy without being useful to any stakeholder.

## 7. Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## 8. Usage

```bash
python public_safety_ml.py
```

Running the script will:
1. Print exploration and missing-value diagnostics to the console.
2. Generate and save the five visualization PNGs listed above.
3. Print the full modelling pipeline output — feature selection reasoning, train/test split summary, and class-balance verification.

## 9. Course

**COMP247 — Supervised Learning**
Centennial College — Semester 4
