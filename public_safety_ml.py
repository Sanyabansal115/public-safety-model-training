import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# 1. LOAD DATA
df = pd.read_csv('KSI_data.csv', encoding='utf-8-sig')
print("1. Loading Data")
print(f"Original dataset shape: {df.shape}")
print(f"Total records: {df.shape[0]}, Total columns: {df.shape[1]}")

# 2. TARGET VARIABLE PREPARATION
print("2. TARGET VARIABLE PREPARATION")

print(f"\nOriginal ACCLASS distribution:\n{df['ACCLASS'].value_counts(dropna=False)}")

# Drop the 1 row with null ACCLASS — cannot train on an unlabeled target
df = df.dropna(subset=['ACCLASS'])

# Merge "Property Damage O" (18 rows) into "Non-Fatal Injury"
# Justification: Only 18 rows — too few to be a standalone class.
# Project goal is binary classification: fatal vs non-fatal.
df['ACCLASS'] = df['ACCLASS'].replace('Property Damage O', 'Non-Fatal Injury')

# Encode: Fatal = 1, Non-Fatal = 0
df['ACCLASS_BINARY'] = (df['ACCLASS'] == 'Fatal').astype(int)

print(f"\nBinary target distribution:")
print(f"  Non-Fatal (0): {(df['ACCLASS_BINARY'] == 0).sum()}")
print(f"  Fatal    (1): {(df['ACCLASS_BINARY'] == 1).sum()}")
print(f"  Imbalance ratio: 1:{(df['ACCLASS_BINARY'] == 0).sum() // (df['ACCLASS_BINARY'] == 1).sum()}")

# 3. DATA TRANSFORMATIONS — HANDLING MISSING VALUES
print("3. HANDLING MISSING DATA")

# Show missing values before treatment
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)
print("\nMissing values summary (before treatment):")
print(missing_report[missing_report['Missing Count'] > 0])

# TIER 1: Binary flag columns — fill NaN with "No"
# Justification: These are Yes/NaN indicator flags. NaN means the condition
# does not apply to this record (e.g., PEDESTRIAN=NaN when no pedestrian was
# involved). These are implicit negatives, NOT unknown values.
binary_flag_cols = [
    'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
    'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER',
    'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY'
]

print("\nTIER 1: Binary flag columns — NaN → 'No' (implicit negatives)")
for col in binary_flag_cols:
    before = df[col].isnull().sum()
    df[col] = df[col].fillna('No')
    print(f"  {col}: filled {before} NaN values with 'No'")

# TIER 2: High-missingness columns (>40%) — DROP ──
# Justification: These are conditional fields populated only for specific
# involvement types (e.g., pedestrian details exist only when PEDESTRIAN=Yes).
# Extreme sparsity adds noise rather than signal. The binary flag columns
# already capture whether that involvement type was present.
# INJURY and FATAL_NO are post-incident outcomes — using them to predict
# fatality is data leakage (the model would "cheat" by seeing outcome info).
high_missing_cols = [
    'PEDTYPE', 'PEDACT', 'PEDCOND',
    'CYCLISTYPE', 'CYCACT', 'CYCCOND',
    'FATAL_NO',                             
    'OFFSET',                               
    'INJURY',                               
    'MANOEUVER',                            
    'DRIVACT',                              
    'DRIVCOND',                             
]

print(f"\nTIER 2: Dropping columns with >40% missing or data leakage")
for col in high_missing_cols:
    pct = (df[col].isnull().sum() / len(df) * 100)
    print(f"  Dropping '{col}' — {pct:.1f}% missing")
df = df.drop(columns=high_missing_cols)

# TIER 3: Low-missingness categorical columns (<5%) — mode imputation ──
# Justification: Missing rates are small enough (<3%) that filling with
# the most common value introduces negligible bias. Mode is the standard
# imputation method for categorical features.
low_missing_cols = ['ROAD_CLASS', 'TRAFFCTL', 'VISIBILITY', 'LIGHT',
                    'RDSFCOND', 'ACCLOC', 'IMPACTYPE', 'INVTYPE',
                    'INITDIR', 'VEHTYPE', 'DISTRICT']

print(f"\nTIER 3: Mode imputation for low-missingness columns (<5%)")
for col in low_missing_cols:
    if col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  {col}: filled {n_missing} NaN with mode '{mode_val}'")

print(f"\nRemaining missing values in working columns:")
remaining = df.isnull().sum()
remaining = remaining[remaining > 0]
if len(remaining) == 0:
    print("  No missing values remain")
else:
    print(remaining)


# 4. FEATURE SELECTION — WITH JUSTIFICATION
print("\n4. FEATURE SELECTION")

# Columns to DROP
drop_cols = {
    'OBJECTID':             'Row identifier — no predictive value',
    'INDEX':                'Row identifier — no predictive value',
    'ACCNUM':               'Accident number — unique ID, no predictive value',
    'STREET1':              'Too many unique values (~4600); location already captured by HOOD_158 and DISTRICT',
    'STREET2':              'Too many unique values; location already captured by HOOD_158 and DISTRICT',
    'ACCLASS':              'Original target label — replaced by binary ACCLASS_BINARY',
    'NEIGHBOURHOOD_158':    'Text name that duplicates numeric HOOD_158 (redundant)',
    'NEIGHBOURHOOD_140':    'Text name that duplicates numeric HOOD_140 (redundant)',
    'HOOD_140':             'Redundant with HOOD_158 — both encode neighbourhood, keeping one',
    'x':                    'Projected x-coordinate — redundant with LATITUDE',
    'y':                    'Projected y-coordinate — redundant with LONGITUDE',
    'DATE':                 'Raw date string — useful info extracted into HOUR feature from TIME',
}

print("\nColumns DROPPED:")
for col, reason in drop_cols.items():
    print(f"  ✗ {col:25s} → {reason}")

df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Columns KEPT
kept_cols = {
    'TIME':            'Hour of day — captures temporal patterns (night vs day fatality risk)',
    'ROAD_CLASS':      'Road type — arterials vs collectors have different fatality profiles',
    'DISTRICT':        'Geographic district — captures area-level risk differences',
    'LATITUDE':        'Geographic coordinate — spatial clustering of fatal collisions',
    'LONGITUDE':       'Geographic coordinate — spatial clustering of fatal collisions',
    'ACCLOC':          'Accident location type (intersection, mid-block) — structural risk factor',
    'TRAFFCTL':        'Traffic control present — signals vs uncontrolled affects severity',
    'VISIBILITY':      'Weather visibility — rain/snow/fog affect crash severity',
    'LIGHT':           'Lighting condition — dark conditions correlate with higher fatality',
    'RDSFCOND':        'Road surface — wet/icy roads affect collision outcomes',
    'IMPACTYPE':       'Impact type — head-on vs sideswipe have very different fatality rates',
    'INVTYPE':         'Involvement type — pedestrians have higher fatality risk than drivers',
    'INVAGE':          'Age group of person involved — elderly more vulnerable',
    'INITDIR':         'Initial direction of travel — directional collision patterns',
    'VEHTYPE':         'Vehicle type — trucks vs cars produce different severity',
    'HOOD_158':        'Neighbourhood code — local area risk proxy',
    'DIVISION':        'Police division — geographic/demographic risk proxy',
    'PEDESTRIAN':      'Binary flag — pedestrian involvement strongly predicts fatality',
    'CYCLIST':         'Binary flag — cyclist involvement',
    'AUTOMOBILE':      'Binary flag — automobile involvement',
    'MOTORCYCLE':      'Binary flag — motorcycle crashes have high fatality',
    'TRUCK':           'Binary flag — truck involvement increases severity',
    'TRSN_CITY_VEH':   'Binary flag — transit/city vehicle involvement',
    'EMERG_VEH':       'Binary flag — emergency vehicle involvement',
    'PASSENGER':       'Binary flag — passenger presence',
    'SPEEDING':        'Binary flag — speeding is a top fatality predictor',
    'AG_DRIV':         'Binary flag — aggressive driving behaviour',
    'REDLIGHT':        'Binary flag — running red lights',
    'ALCOHOL':         'Binary flag — alcohol involvement strongly predicts fatality',
    'DISABILITY':      'Binary flag — disability involvement',
}

print(f"\nColumns KEPT:")
for col, reason in kept_cols.items():
    print(f"  ✓ {col:25s} → {reason}")


# 5. DATA TRANSFORMATIONS — CATEGORICAL ENCODING & FEATURE ENGINEERING
print("\n5. CATEGORICAL DATA MANAGEMENT & FEATURE ENGINEERING")

# 5a. Extract HOUR from TIME
# TIME is stored as integer (e.g. 1430 = 2:30 PM). Extract hour for a
# cleaner numeric feature.
df['HOUR'] = df['TIME'] // 100
df = df.drop(columns=['TIME'])
print(f"  Extracted HOUR from TIME (range: {df['HOUR'].min()} – {df['HOUR'].max()})")

# 5b. Handle INVAGE "unknown"
# ~14% of INVAGE values are "unknown". Keeping as its own category because
# the unknownness may correlate with outcome (hit-and-runs, unidentified).
print(f" INVAGE 'unknown' entries: {(df['INVAGE'] == 'unknown').sum()} — kept as category")

# 5c. Group rare categories (<1% frequency) into "Other"
# This prevents one-hot encoding from creating many near-empty columns
# that add dimensionality without predictive signal.
def group_rare(series, threshold=0.01):
    """Replace categories appearing below threshold fraction with 'Other'."""
    freq = series.value_counts(normalize=True)
    rare = freq[freq < threshold].index
    return series.replace(rare, 'Other')

rare_grouping_cols = ['ROAD_CLASS', 'LIGHT', 'RDSFCOND', 'VEHTYPE',
                      'INVTYPE', 'TRAFFCTL', 'VISIBILITY']

for col in rare_grouping_cols:
    if col in df.columns:
        before = df[col].nunique()
        df[col] = group_rare(df[col])
        after = df[col].nunique()
        if before != after:
            print(f"  {col}: grouped rare categories ({before} → {after} unique values)")

# 5d. Convert binary flag strings to 0/1
for col in binary_flag_cols:
    if col in df.columns:
        df[col] = (df[col] == 'Yes').astype(int)
print(f"  Encoded {len(binary_flag_cols)} binary flag columns: 'Yes'→1, 'No'→0")

# 5e. Dataset overview after transformations
print(f"\nDataset shape after transformations: {df.shape}")
print(f"Feature types:\n{df.dtypes.value_counts()}")


# 6. TRAIN / TEST SPLIT (STRATIFIED)
print("\n6. TRAIN / TEST SPLIT")

X = df.drop(columns=['ACCLASS_BINARY'])
y = df['ACCLASS_BINARY']

# Identify column types for ColumnTransformer
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# 80/20 split with stratification
# stratify=y ensures both sets preserve the original Fatal/Non-Fatal ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"  Non-Fatal (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Fatal    (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
print(f"\nTest set: {X_test.shape[0]} samples")
print(f"  Non-Fatal (0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"  Fatal    (1): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")
print(f"\n  Stratification verified — both sets have ~{(y==1).mean()*100:.1f}% Fatal class")


# 7. PREPROCESSING PIPELINE (ColumnTransformer)
print("\n7. PREPROCESSING PIPELINE")

# Numeric pipeline: impute remaining edge cases with median, then standardize.
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute with mode, then one-hot encode.
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer applies each pipeline to its respective column group
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='drop'
)

# Fit on TRAINING data only, then transform both sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
encoded_cat_names = preprocessor.named_transformers_['cat']['encoder'] \
    .get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(encoded_cat_names)

print(f"  Numeric pipeline:     SimpleImputer(median) → StandardScaler")
print(f"  Categorical pipeline: SimpleImputer(mode) → OneHotEncoder")
print(f"\n  Features before encoding: {X.shape[1]}")
print(f"  Features after encoding:  {X_train_processed.shape[1]}")
print(f"  Training matrix shape:    {X_train_processed.shape}")
print(f"  Test matrix shape:        {X_test_processed.shape}")

# 8. MANAGING IMBALANCED CLASSES (SMOTE)
print("\n8. MANAGING IMBALANCED CLASSES — SMOTE")

print(f"\nClass distribution BEFORE SMOTE (training set):")
print(f"  {Counter(y_train)}")

# SMOTE applied ONLY to training data — test data must remain untouched for
# honest evaluation.
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"\nClass distribution AFTER SMOTE (training set):")
print(f"  {Counter(y_train_resampled)}")
print(f"\n  Resampled training shape: {X_train_resampled.shape}")
print(f"  Test set remains untouched: {X_test_processed.shape}")


# SUMMARY
print("\nDELIVERABLE 2 — SUMMARY")
print(f"""
  Original records:          {18957}
  Records after cleaning:    {df.shape[0]}
  Features selected:         {X.shape[1]}
  Features after encoding:   {X_train_processed.shape[1]}

  Target: ACCLASS_BINARY (Fatal=1, Non-Fatal=0)

  Missing data strategy:
    Tier 1 — Binary flags: NaN filled with 'No' (implicit negatives)
    Tier 2 — >40% missing or data leakage: columns dropped
    Tier 3 — <5% missing: mode imputation

  Categorical encoding:
    Binary flags → 0/1 integers
    Multi-category columns → OneHotEncoder (via pipeline)
    Rare categories (<1%) → grouped into 'Other'

  Normalization: StandardScaler on numeric features (via pipeline)

  Train/Test split: 80/20, stratified by target
    Train: {X_train.shape[0]} samples
    Test:  {X_test.shape[0]} samples

  Class imbalance: SMOTE on training data only
    Before: {dict(Counter(y_train))}
    After:  {dict(Counter(y_train_resampled))}

  Pipeline: ColumnTransformer(
    numeric     → SimpleImputer(median) → StandardScaler
    categorical → SimpleImputer(mode)   → OneHotEncoder
  )
""")