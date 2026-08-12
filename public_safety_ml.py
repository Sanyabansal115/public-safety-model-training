import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit


df = pd.read_csv("KSI_data.csv")
print(df.head())

print(df.shape)

df.info()

print(df.isnull().sum())

summary_df = pd.DataFrame(
    {
        "Data Type": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2),
        "Unique Values": df.nunique(),
    }
)
print(summary_df)

# ranges and values of elements
print("Numeric Statistical Assessment")
print(df.describe().T)

print("Categorical Statistical Assessment")
print(df.describe(include="object").T)

# The missing-value stage is actually the trickiest part of this dataset, because
# most of the "missingness" here isn't random — it means something specific.
# Treating it carelessly (e.g. one global dropna() or filling everything with
# mode) will wreck other columns.

# Handling Missing Values:

# Group 1 — Binary involvement flags (huge missing %, but missing = "No")
# PEDESTRIAN, CYCLIST, AUTOMOBILE, MOTORCYCLE, TRUCK, TRSN_CITY_VEH, EMERG_VEH,
# PASSENGER, SPEEDING, AG_DRIV, REDLIGHT, ALCOHOL, DISABILITY
# These columns only contain the value "Yes" when true, and NaN otherwise —
# there's no "No" recorded. 95%+ missing on ALCOHOL/DISABILITY doesn't mean
# "bad data," it means alcohol/disability wasn't a factor in 95% of collisions.
# Fix: fillna("No"), not drop, not mode-impute.
group1_cols = [
    "PEDESTRIAN",
    "CYCLIST",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "TRUCK",
    "TRSN_CITY_VEH",
    "EMERG_VEH",
    "PASSENGER",
    "SPEEDING",
    "AG_DRIV",
    "REDLIGHT",
    "ALCOHOL",
    "DISABILITY",
]

df[group1_cols] = df[group1_cols].fillna("No")
print(df[group1_cols].isnull().sum())

print(df["ALCOHOL"].value_counts())
print("\n", df["AG_DRIV"].value_counts())

# Group 2 — Role-conditional attributes (missing = not applicable to that record)
# PEDTYPE, PEDACT, PEDCOND only apply when the involved person is a pedestrian;
# CYCLISTYPE, CYCACT, CYCCOND only apply to cyclists.
# Their ~83-96% missingness is because most records aren't pedestrians/cyclists
# at all.
# Fix: fillna("Not Applicable") — dropping rows would delete almost every
# non-pedestrian/non-cyclist record.
group2_cols = [
    # Pedestrian-specific features
    "PEDTYPE",
    "PEDACT",
    "PEDCOND",
    # Cyclist-specific features
    "CYCLISTYPE",
    "CYCACT",
    "CYCCOND",
]

df[group2_cols] = df[group2_cols].fillna("Not Applicable")
print(df[group2_cols].isnull().sum())

print(df["PEDACT"].value_counts())
print("\n", df["CYCACT"].value_counts())

# Group 3 — Event-conditional FATAL_NO (95.4% missing)
# Only has a value for fatal injuries — it's a sequence number for fatalities
# in a multi-fatality collision.
# Missing = not fatal. fillna(0) or leave as "N/A" flag, not statistical
# imputation.
# Fix: Fill NaN Values with 0
df["FATAL_NO"] = df["FATAL_NO"].fillna(0).astype(int)
print("Missing count in FATAL_NO:", df["FATAL_NO"].isnull().sum())

print(df["FATAL_NO"].value_counts().head(10))

# Group 4 — Genuinely missing/incomplete records (true missing data)
# ACCLOC (28.8%), INITDIR (27.8%), ROAD_CLASS (2.6%), DISTRICT (1.2%),
# TRAFFCTL (0.4%), VISIBILITY, RDSFCOND, LIGHT, IMPACTYPE, INVTYPE, ACCLASS
# (all <1%) — these are cases where the report just wasn't filled in
# completely. This is only genuine "missing data" group.
# Fix: The strategy is split into two parts:
#   A - dropping negligible missing rows (<3%)
#   B - imputing medium missingness (~28%) as "Unknown"
group3A_cols = [
    "ACCLASS",
    "LIGHT",
    "VISIBILITY",
    "IMPACTYPE",
    "RDSFCOND",
    "TRAFFCTL",
    "DISTRICT",
    "ROAD_CLASS",
]

group3B_cols = ["ACCLOC", "INITDIR"]

df = df.dropna(subset=group3A_cols)
df[group3B_cols] = df[group3B_cols].fillna("Unkown")

print("Negligible Missing Rows\n", df[group3A_cols].isnull().sum())
print("\nMedium Missing Rows\n", df[group3B_cols].isnull().sum())

# Group 5 — Identifiers, not analysis variables
# OFFSET (79.8% missing), STREET2 (9%), ACCNUM (26%), VEHTYPE (18.4%)
# These are location/ID descriptors.
# ACCNUM missingness usually reflects records where an official number
# wasn't assigned
group5_cols = [
    "OFFSET",
    "STREET2",
    "VEHTYPE"]

df[group5_cols] = df[group5_cols].fillna("Unkown")

missing_accnum_mask = df["ACCNUM"].isnull()
df.loc[missing_accnum_mask, "ACCNUM"] = -1 * (df.loc[missing_accnum_mask].index + 1)

print(df[group5_cols].isnull().sum())
print("ACCNUM ", df["ACCNUM"].isnull().sum())

print(df.isnull().sum())

#Group-6
# Check whether missingness in [[MANOEUVER, DRIVACT, DRIVCOND, INJURY] columns lines up with INVTYPE categories
print(df.groupby("INVTYPE")["MANOEUVER"].apply(lambda x: x.isnull().mean()))
print(df.groupby("INVTYPE")["DRIVACT"].apply(lambda x: x.isnull().mean()))
print(df.groupby("INVTYPE")["DRIVCOND"].apply(lambda x: x.isnull().mean()))
print(df.groupby("INVTYPE")["INJURY"].apply(lambda x: x.isnull().mean()))

#Result: 
   # Missingness by INVTYPE Summary
   # MANOEUVER: Role-driven. Almost fully populated for vehicle operators (~0–0.4% missing) and almost entirely missing for non-operators (96–100%).
   # DRIVACT / DRIVCOND: Driver-specific. Populated only for drivers (~0.3–1.1% missing); 100% missing for all other roles.
   #INJURY: Mixed mechanism. Bystanders are ~100% missing (Not Applicable). Scene-present roles show variable missingness (e.g., Driver 63%, Passenger 35%), which indicates unrecorded non-injuries rather than non-applicability (None).

   #Action: Fill MANOEUVER, DRIVACT, and DRIVCOND uniformly with "Not Applicable". Fill INJURY conditionally using an INVTYPE mask ("None" for scene-present roles, "Not Applicable" for bystanders).

print(df["INJURY"].value_counts(dropna=False))


# INVTYPE — 16 missing, negligible, drop
df = df.dropna(subset=["INVTYPE"])

# MANOEUVER, DRIVACT, DRIVCOND — clean bimodal split by role (confirmed via crosstab)
# Missing = this person wasn't operating a vehicle, so the field doesn't apply
df[["MANOEUVER", "DRIVACT", "DRIVCOND"]] = df[["MANOEUVER", "DRIVACT", "DRIVCOND"]].fillna("Not Applicable")

# INJURY — NOT bimodal (Driver 63% missing, Passenger 35%, Truck Driver 83%)
# For roles physically present at the scene, missing likely means "not hurt"
# For bystander roles (Witness, Vehicle Owner, etc.), missing means the field never applied
# These are different facts, so they get different labels
present_roles = ["Driver", "Passenger", "Truck Driver", "Motorcycle Driver",
                  "Motorcycle Passenger", "Cyclist", "Cyclist Passenger",
                  "Moped Driver", "Moped Passenger", "In-Line Skater", "Wheelchair"]

mask_present = df["INVTYPE"].isin(present_roles)

df.loc[mask_present, "INJURY"] = df.loc[mask_present, "INJURY"].fillna("None")
df.loc[~mask_present, "INJURY"] = df.loc[~mask_present, "INJURY"].fillna("Not Applicable")

print(df.isnull().sum())

df["DATE_PARSED"] = pd.to_datetime(df["DATE"])
df["YEAR"] = df["DATE_PARSED"].dt.year
df["HOUR"] = df["TIME"] // 100


# =============================================================================
# DELIVERABLE 1B — DATA VISUALIZATION
# =============================================================================
# Exploratory charts built on the cleaned frame from the exploration section
# above. No modelling transformations (encoding, scaling, feature drops) have
# happened yet, so these plots reflect the raw-but-cleaned KSI data.

# 1. Collisions per year, split by severity
yearly = df.groupby(["YEAR", "ACCLASS"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 5.5))
yearly.plot(kind="bar", stacked=True, ax=ax,
            color=["#d62728", "#1f77b4", "#7f7f7f"])
ax.set_title("Killed or Seriously Injured Collisions by Year and Severity")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Collisions")
ax.legend(title="Severity")
plt.tight_layout()
plt.savefig("1_collisions_by_year_severity.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. Collisions by hour of day
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x="HOUR", data=df, hue="ACCLASS", ax=ax,
              palette={"Fatal": "#d62728", "Non-Fatal Injury": "#1f77b4",
                       "Property Damage O": "#7f7f7f"})
ax.set_title("Collisions by Hour of Day")
ax.set_xlabel("Hour (24h)")
ax.set_ylabel("Number of Collisions")
plt.tight_layout()
plt.savefig("2_collisions_by_hour.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Contributing factors (Yes counts across binary flags)
factor_cols = ["SPEEDING", "AG_DRIV", "REDLIGHT", "ALCOHOL", "DISABILITY"]
factor_counts = {c: (df[c] == "Yes").sum() for c in factor_cols}
factor_series = pd.Series(factor_counts).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
factor_series.plot(kind="barh", ax=ax, color="#c44e52")
ax.set_title("Collisions Involving Each Contributing Factor")
ax.set_xlabel("Number of Collisions")
plt.tight_layout()
plt.savefig("3_contributing_factors.png", dpi=300, bbox_inches="tight")
plt.show()

# 4. Correlation heatmap of binary contributing/involvement factors
bin_cols = ["PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK",
            "SPEEDING", "AG_DRIV", "REDLIGHT", "ALCOHOL", "DISABILITY"]
bin_df = df[bin_cols].apply(lambda s: (s == "Yes").astype(int))
corr = bin_df.corr()
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            square=True, cbar_kws={"label": "Correlation"})
ax.set_title("Correlation Between Contributing Factors")
plt.tight_layout()
plt.savefig("4_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Top 10 neighbourhoods by collision count
top_nbhd = df["NEIGHBOURHOOD_140"].value_counts().nlargest(10).sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
top_nbhd.plot(kind="barh", ax=ax, color="#dd8452")
ax.set_title("Top 10 Neighbourhoods by Collision Count")
ax.set_xlabel("Number of Collisions")
plt.tight_layout()
plt.savefig("5_top_neighbourhoods.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# DELIVERABLE 2 — DATA MODELLING
# =============================================================================
# Everything above is Deliverable 1 (data exploration + missing-value strategy).
# This section builds on that cleaned frame. It deliberately does NOT reload the
# CSV and does NOT repeat any imputation — all of that already happened above,
# and running it twice would either double-transform values or silently undo the
# group-by-group decisions that were justified in Deliverable 1.

import numpy as np
from collections import Counter

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


# 1. WORK ON A COPY OF THE EXPLORED DATA
print("\n1. PREPARING THE MODELLING FRAME")

# model_df is the only frame the modelling stage touches. df is left exactly as
# Deliverable 1 produced it, so the exploration results stay reproducible and
# re-running any cell above cannot be affected by modelling transformations.
model_df = df.copy()
print(f"Rows entering modelling: {model_df.shape[0]}, columns: {model_df.shape[1]}")


# 2. TARGET VARIABLE PREPARATION
print("\n2. TARGET VARIABLE PREPARATION")

print(f"\nACCLASS distribution after cleaning:\n{model_df['ACCLASS'].value_counts(dropna=False)}")

# No dropna() on ACCLASS here — the single unlabeled row was already removed by
# the Group 4A dropna() above, so repeating it would be a no-op at best.

# Merge "Property Damage O" (18 rows) into "Non-Fatal Injury"
# Justification: only 18 rows — too few to be a standalone class, and the
# project goal is binary classification: fatal vs non-fatal.
model_df["ACCLASS"] = model_df["ACCLASS"].replace("Property Damage O", "Non-Fatal Injury")

# Encode: Fatal = 1, Non-Fatal = 0
model_df["ACCLASS_BINARY"] = (model_df["ACCLASS"] == "Fatal").astype(int)

n_nonfatal = (model_df["ACCLASS_BINARY"] == 0).sum()
n_fatal = (model_df["ACCLASS_BINARY"] == 1).sum()
print("\nBinary target distribution:")
print(f"  Non-Fatal (0): {n_nonfatal}")
print(f"  Fatal     (1): {n_fatal}")
print(f"  Imbalance ratio: 1:{n_nonfatal // n_fatal}")


# 3. MISSING DATA — VERIFICATION ONLY (already handled in Deliverable 1)
print("\n3. MISSING DATA VERIFICATION")

# The Group 1-6 strategy above resolved every column, so this is a guard rail
# rather than another imputation pass. If a column ever shows up here, it means
# the exploration stage missed it — fix it up there, not down here.
remaining = model_df.isnull().sum()
remaining = remaining[remaining > 0]
if remaining.empty:
    print("  No missing values remain — no further imputation applied.")
else:
    print("  Columns still missing values (revisit Deliverable 1):")
    print(remaining)


# 4. FEATURE SELECTION — WITH JUSTIFICATION
print("\n4. FEATURE SELECTION")

# 4a. Post-incident outcome columns — DROP (data leakage)
# These describe what happened *after* the collision. Using them to predict
# fatality would let the model cheat: FATAL_NO is only populated for fatalities
# and INJURY records the severity outcome directly.
leakage_cols = {
    "INJURY": "Injury severity is the outcome itself — direct target leakage",
    "FATAL_NO": "Only populated for fatal collisions — perfect leakage of the target",
}

# 4b. Sparse role-conditional columns — DROP
# These were filled with "Not Applicable" above so the missingness could be
# explained, but 83-96% of the rows carry that single placeholder. The binary
# involvement flags (PEDESTRIAN, CYCLIST) already capture the same information
# with none of the sparsity.
sparse_cols = {col: "83-96% 'Not Applicable' — signal already captured by the PEDESTRIAN/CYCLIST flags"
               for col in group2_cols}
sparse_cols.update({
    "MANOEUVER": "Populated only for vehicle operators — mostly 'Not Applicable'",
    "DRIVACT": "Driver-only field — mostly 'Not Applicable'",
    "DRIVCOND": "Driver-only field — mostly 'Not Applicable'",
    "OFFSET": "79.8% placeholder — free-text location offset, no predictive value",
})

# 4c. Identifiers and redundant columns — DROP
# ACCNUM is deliberately NOT in this dict — Section 6 pulls it out separately
# as the grouping key for the train/test split, then drops it from X after.
identifier_cols = {
    "OBJECTID": "Row identifier — no predictive value",
    "INDEX": "Row identifier — no predictive value",
    "STREET1": "~4600 unique values; location already captured by HOOD_158 and DISTRICT",
    "STREET2": "Too many unique values; location already captured by HOOD_158 and DISTRICT",
    "ACCLASS": "Original target label — replaced by binary ACCLASS_BINARY",
    "NEIGHBOURHOOD_158": "Text name that duplicates numeric HOOD_158 (redundant)",
    "NEIGHBOURHOOD_140": "Text name that duplicates numeric HOOD_140 (redundant)",
    "HOOD_140": "Redundant with HOOD_158 — both encode neighbourhood, keeping one",
    "x": "Projected x-coordinate — redundant with LONGITUDE",
    "y": "Projected y-coordinate — redundant with LATITUDE",
    "DATE": "Raw date string — the usable temporal signal is extracted as HOUR from TIME",
    "DATE_PARSED": "Intermediate column used only to compute YEAR — not a feature",
    "YEAR": "Used for the exploration chart, not carried into modelling as a feature",
}

drop_cols = {**leakage_cols, **sparse_cols, **identifier_cols}

print("\nColumns DROPPED:")
for col, reason in drop_cols.items():
    print(f"  x {col:20s} -> {reason}")

model_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])

# Columns KEPT — justification for each retained feature
kept_cols = {
    "TIME": "Converted to HOUR below — captures night vs day fatality risk",
    "ROAD_CLASS": "Road type — arterials vs collectors have different fatality profiles",
    "DISTRICT": "Geographic district — captures area-level risk differences",
    "LATITUDE": "Geographic coordinate — spatial clustering of fatal collisions",
    "LONGITUDE": "Geographic coordinate — spatial clustering of fatal collisions",
    "ACCLOC": "Accident location type (intersection, mid-block) — structural risk factor",
    "TRAFFCTL": "Traffic control present — signals vs uncontrolled affects severity",
    "VISIBILITY": "Weather visibility — rain/snow/fog affect crash severity",
    "LIGHT": "Lighting condition — dark conditions correlate with higher fatality",
    "RDSFCOND": "Road surface — wet/icy roads affect collision outcomes",
    "IMPACTYPE": "Impact type — head-on vs sideswipe have very different fatality rates",
    "INVTYPE": "Involvement type — pedestrians have higher fatality risk than drivers",
    "INVAGE": "Age group of person involved — elderly are more vulnerable",
    "INITDIR": "Initial direction of travel — directional collision patterns",
    "VEHTYPE": "Vehicle type — trucks vs cars produce different severity",
    "HOOD_158": "Neighbourhood code — local area risk proxy",
    "DIVISION": "Police division — geographic/demographic risk proxy",
    "PEDESTRIAN": "Binary flag — pedestrian involvement strongly predicts fatality",
    "CYCLIST": "Binary flag — cyclist involvement",
    "AUTOMOBILE": "Binary flag — automobile involvement",
    "MOTORCYCLE": "Binary flag — motorcycle crashes have high fatality",
    "TRUCK": "Binary flag — truck involvement increases severity",
    "TRSN_CITY_VEH": "Binary flag — transit/city vehicle involvement",
    "EMERG_VEH": "Binary flag — emergency vehicle involvement",
    "PASSENGER": "Binary flag — passenger presence",
    "SPEEDING": "Binary flag — speeding is a top fatality predictor",
    "AG_DRIV": "Binary flag — aggressive driving behaviour",
    "REDLIGHT": "Binary flag — running red lights",
    "ALCOHOL": "Binary flag — alcohol involvement strongly predicts fatality",
    "DISABILITY": "Binary flag — disability involvement",
}

print("\nColumns KEPT:")
for col, reason in kept_cols.items():
    print(f"  + {col:20s} -> {reason}")


# 5. CATEGORICAL DATA MANAGEMENT & FEATURE ENGINEERING
print("\n5. CATEGORICAL DATA MANAGEMENT & FEATURE ENGINEERING")

# 5a. Extract HOUR from TIME
# TIME is stored as an integer (e.g. 1430 = 2:30 PM), so integer-divide by 100.
# Guarded by a column check so re-running this section cannot divide twice.
if "TIME" in model_df.columns:
    model_df["HOUR"] = model_df["TIME"] // 100
    model_df = model_df.drop(columns=["TIME"])
print(f"  Extracted HOUR from TIME (range: {model_df['HOUR'].min()} - {model_df['HOUR'].max()})")

# 5b. Handle INVAGE "unknown"
# ~14% of INVAGE values are "unknown". Kept as its own category because the
# unknownness itself may correlate with outcome (hit-and-runs, unidentified).
print(f"  INVAGE 'unknown' entries: {(model_df['INVAGE'] == 'unknown').sum()} — kept as a category")


# 5c. Group rare categories (<1% frequency) into "Other"
# Prevents one-hot encoding from creating many near-empty columns that add
# dimensionality without predictive signal.
def group_rare(series, threshold=0.01):
    """Replace categories appearing below `threshold` fraction with 'Other'."""
    freq = series.value_counts(normalize=True)
    rare = freq[freq < threshold].index
    return series.where(~series.isin(rare), "Other")


rare_grouping_cols = ["ROAD_CLASS", "LIGHT", "RDSFCOND", "VEHTYPE",
                      "INVTYPE", "TRAFFCTL", "VISIBILITY"]

# HOOD_158 is intentionally excluded — with 159 neighbourhoods every single one
# sits below 1%, so grouping would collapse the whole column into "Other".
for col in rare_grouping_cols:
    before = model_df[col].nunique()
    model_df[col] = group_rare(model_df[col])
    after = model_df[col].nunique()
    if before != after:
        print(f"  {col}: grouped rare categories ({before} -> {after} unique values)")

# 5d. Convert the Group 1 involvement flags from strings to 0/1
# group1_cols is reused from Deliverable 1 rather than redeclared, so the two
# stages can never drift apart. The comparison against "Yes" is idempotent.
binary_flag_cols = group1_cols
for col in binary_flag_cols:
    model_df[col] = (model_df[col] == "Yes").astype(int)
print(f"  Encoded {len(binary_flag_cols)} binary flag columns: 'Yes'->1, 'No'->0")

print(f"\nModelling frame shape after transformations: {model_df.shape}")
print(f"Feature types:\n{model_df.dtypes.value_counts()}")


# 6. TRAIN / TEST SPLIT (GROUPED BY CRASH, NOT STRATIFIED)
print("\n6. TRAIN / TEST SPLIT")

X = model_df.drop(columns=["ACCLASS_BINARY"])
y = model_df["ACCLASS_BINARY"]

# Extract crash-level groups BEFORE dropping ACCNUM from X. Each ACCNUM now
# maps to exactly one real crash (Fix 2, above) — no more shared -1 mega-group.
groups = X["ACCNUM"]

# Now drop ACCNUM from X — it's an ID, not a predictive feature
X = X.drop(columns=["ACCNUM"])

# Identify column types for the ColumnTransformer (done once, after the drop,
# so numeric_features/categorical_features never include ACCNUM)
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Extract crash-level groups BEFORE dropping ACCNUM from X
groups = X["ACCNUM"]

# Now drop ACCNUM from X — it's not a predictive feature
X = X.drop(columns=["ACCNUM"])

# Re-identify column types AFTER dropping ACCNUM
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Split by crash ID — whole crashes stay together in train OR test, never split
# across both. GroupShuffleSplit does NOT stratify by y — it only respects
# groups — so the Fatal ratio below is a result, not a guarantee.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]



print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"  Non-Fatal (0): {(y_train == 0).sum()} ({(y_train == 0).mean() * 100:.1f}%)")
print(f"  Fatal     (1): {(y_train == 1).sum()} ({(y_train == 1).mean() * 100:.1f}%)")
print(f"\nTest set: {X_test.shape[0]} samples")
print(f"  Non-Fatal (0): {(y_test == 0).sum()} ({(y_test == 0).mean() * 100:.1f}%)")
print(f"  Fatal     (1): {(y_test == 1).sum()} ({(y_test == 1).mean() * 100:.1f}%)")
print(f"\n  Split by ACCNUM (GroupShuffleSplit), not stratified — Fatal ratio")
print(f"  above is a result of the crash-level split, not an enforced target.")
print(f"  Overall Fatal rate in the full dataset: {(y == 1).mean() * 100:.1f}%")


# 7. PREPROCESSING PIPELINE (ColumnTransformer)
print("\n7. PREPROCESSING PIPELINE")

# Numeric pipeline: median-impute any edge case, then standardize.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline: mode-impute, then one-hot encode.
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# The imputers are a safety net for unseen production rows, not a second pass
# over this dataset — step 3 confirmed nothing is missing here.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# Fit on TRAINING data only, then transform both sets — fitting on the full
# dataset would leak test-set statistics into the scaler and encoder.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Feature names after one-hot encoding (used later for feature importance)
encoded_cat_names = preprocessor.named_transformers_["cat"]["encoder"] \
    .get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(encoded_cat_names)

print("  Numeric pipeline:     SimpleImputer(median) -> StandardScaler")
print("  Categorical pipeline: SimpleImputer(mode) -> OneHotEncoder")
print(f"\n  Features before encoding: {X.shape[1]}")
print(f"  Features after encoding:  {X_train_processed.shape[1]}")
print(f"  Training matrix shape:    {X_train_processed.shape}")
print(f"  Test matrix shape:        {X_test_processed.shape}")


# 8. MANAGING IMBALANCED CLASSES (SMOTE)
print("\n8. MANAGING IMBALANCED CLASSES — SMOTE")

print("\nClass distribution BEFORE SMOTE (training set):")
print(f"  {Counter(y_train)}")

# SMOTE is applied ONLY to the training data — resampling the test set would
# inflate the scores and make the evaluation dishonest.
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print("\nClass distribution AFTER SMOTE (training set):")
print(f"  {Counter(y_train_resampled)}")
print(f"\n  Resampled training shape: {X_train_resampled.shape}")
print(f"  Test set left untouched:  {X_test_processed.shape}")


# SUMMARY
print("\nDELIVERABLE 2 — SUMMARY")
print(f"""
  Records after exploration cleaning: {df.shape[0]}
  Records used for modelling:         {model_df.shape[0]}
  Features selected:                  {X.shape[1]}
  Features after encoding:            {X_train_processed.shape[1]}

  Target: ACCLASS_BINARY (Fatal=1, Non-Fatal=0)

  Missing data: fully handled in Deliverable 1 (Groups 1-6); the modelling
    stage only verifies it and never re-imputes.

  Feature selection:
    Leakage dropped     -> INJURY, FATAL_NO
    Sparse dropped      -> pedestrian/cyclist detail columns, MANOEUVER,
                           DRIVACT, DRIVCOND, OFFSET
    Identifiers dropped -> OBJECTID, INDEX, ACCNUM (used as split group first),
                           STREET1/2, DATE, x, y
    Redundant dropped   -> NEIGHBOURHOOD_158/140, HOOD_140, ACCLASS

  Categorical encoding:
    Binary flags          -> 0/1 integers
    Multi-category        -> OneHotEncoder (inside the pipeline)
    Rare categories (<1%) -> grouped into 'Other'

  Normalization: StandardScaler on numeric features (inside the pipeline)

  Train/Test split: 80/20 by crash (ACCNUM), via GroupShuffleSplit — no crash
    is split across train and test
    Train: {X_train.shape[0]} samples
    Test:  {X_test.shape[0]} samples

  Class imbalance: SMOTE on training data only
    Before: {dict(Counter(y_train))}
    After:  {dict(Counter(y_train_resampled))}

  Pipeline: ColumnTransformer(
    numeric     -> SimpleImputer(median) -> StandardScaler
    categorical -> SimpleImputer(mode)   -> OneHotEncoder
  )
""")
