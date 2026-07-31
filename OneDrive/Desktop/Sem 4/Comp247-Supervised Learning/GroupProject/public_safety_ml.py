import pandas as pd


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
df["ACCNUM"] = df["ACCNUM"].fillna(-1)  # temporary placeholder

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

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
 
# df should already be the cleaned dataframe from your group mate's script.
# If running this standalone, uncomment the line below:
# df = pd.read_csv("KSI_cleaned.csv", low_memory=False)
 
df["DATE"] = pd.to_datetime(df["DATE"])
df["YEAR"] = df["DATE"].dt.year
df["MONTH"] = df["DATE"].dt.month
df["HOUR"] = (df["TIME"] // 100).astype(int)
 
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
