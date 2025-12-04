#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:13:57 2025

@author: jfcaetano
"""

import pandas as pd
import numpy as np

# Inputs
FI_IN   = "feature_importance-full-type-opt.csv"
DATA_IN = "Davide-RDKIT.csv"
FI_OUT  = "feature_importance-full-type-class-opt.csv"

fi = pd.read_csv(FI_IN)
data = pd.read_csv(DATA_IN)

def summarize_classes(series: pd.Series) -> str:
    u = pd.unique(series.dropna().astype(str))
    if len(u) == 0:
        return np.nan
    if len(u) == 1:
        return u[0]
    return ";".join(sorted(u))

mol2class = (
    data.groupby("MOLECULE", dropna=False)["SMILES_Class"]
        .apply(summarize_classes)
        .reset_index()
        .rename(columns={"SMILES_Class": "SMILES_Class"})
)


fi_with_class = fi.merge(mol2class, on="MOLECULE", how="left")

fi_with_class.to_csv(FI_OUT, index=False)
print(f"Saved: {FI_OUT}")




# ---------- Config ----------
FI_IN   = "feature_importance-full-type-opt.csv"
DATA_IN = "Davide-RDKIT.csv"
FI_WITH_CLASS_OUT = "feature_importance-full-type-class-opt.csv"
AGG_OUT = "feature_importance_by_SMILESClass-opt.csv"
PIVOT_OUT = "feature_importance_by_SMILESClass-pivot-opt.csv"

fi = pd.read_csv(FI_IN)
data = pd.read_csv(DATA_IN)

req_fi_cols = {"MOLECULE", "Model", "Feature", "Importance"}
missing = req_fi_cols - set(fi.columns)
if missing:
    raise ValueError(f"Missing required columns in {FI_IN}: {missing}")

def summarize_classes(series: pd.Series) -> str:
    u = pd.unique(series.dropna().astype(str))
    if len(u) == 0:
        return np.nan
    if len(u) == 1:
        return u[0]
    # If a molecule appears with multiple classes, keep a stable, readable union
    return ";".join(sorted(u))

mol2class = (
    data.groupby("MOLECULE", dropna=False)["SMILES_Class"]
        .apply(summarize_classes)
        .reset_index()
)

fi_with_class = fi.merge(mol2class, on="MOLECULE", how="left")

# Save the enriched long table (one row per fold/molecule/model/feature)
fi_with_class.to_csv(FI_WITH_CLASS_OUT, index=False)
print(f"[Info] Saved per-row feature importances with SMILES_Class → {FI_WITH_CLASS_OUT}")

# We aggregate multiple rows (across molecules/folds) *within* each (Model, SMILES_Class, Feature)

grp = fi_with_class.groupby(["Model", "SMILES_Class", "Feature"], dropna=False)["Importance"]

agg = grp.agg(
    sum_importance = "sum",     
    mean_importance = "mean",   
    std_importance = "std",     
    n = "count"                 
).reset_index()



total_sum_per_group = agg.groupby(["Model", "SMILES_Class"], dropna=False)["sum_importance"].transform("sum")
agg["sum_share"] = agg["sum_importance"] / total_sum_per_group


total_mean_per_group = agg.groupby(["Model", "SMILES_Class"], dropna=False)["mean_importance"].transform("sum")

agg["mean_share"] = np.where(total_mean_per_group.gt(0), agg["mean_importance"] / total_mean_per_group, np.nan)


for col in ["Type", "Source"]:
    if col in fi_with_class.columns and col not in agg.columns:

        meta = (
            fi_with_class
            .dropna(subset=[col])
            .groupby(["Model", "SMILES_Class", "Feature"], dropna=False)[col]
            .first()
            .reset_index()
        )
        agg = agg.merge(meta, on=["Model", "SMILES_Class", "Feature"], how="left")

# Sort for readability: by Model, Class, then descending by sum_share
agg = agg.sort_values(["Model", "SMILES_Class", "sum_share"], ascending=[True, True, False])
agg.to_csv(AGG_OUT, index=False)
print(f"[Info] Saved aggregated feature importances by SMILES_Class → {AGG_OUT}")

pivot = agg.pivot_table(
    index=["Model", "SMILES_Class"],
    columns="Feature",
    values="sum_share",
    fill_value=0.0,
    aggfunc="first"
)

pivot.to_csv(PIVOT_OUT)
print(f"[Info] Saved pivot (Model×Class × Feature) of normalized importance (sum_share) → {PIVOT_OUT}")



"""
Build an overall (across all SMILES classes / folds) feature-importance table,
including standard deviation of FI values.

"""


FI_IN  = "feature_importance-full-type-opt.csv"
FI_OUT = "feature_importance_overall-opt.csv"
AGGREGATION = "mean"

fi = pd.read_csv(FI_IN)

# sanity check
required = {"Model", "Feature", "Importance"}
missing = required - set(fi.columns)
if missing:
    raise ValueError(f"Missing required columns in {FI_IN}: {missing}")

for col in ["Type", "Source"]:
    if col not in fi.columns:
        fi[col] = pd.NA

def first_non_null(s: pd.Series):
    return s.dropna().iloc[0] if s.dropna().size else pd.NA

group_cols = ["Model", "Feature"]

if AGGREGATION == "sum":
    agg = fi.groupby(group_cols, dropna=False)["Importance"].agg(["sum", "std"])
    agg = agg.rename(columns={"sum": "Importance", "std": "Importance_std"})
elif AGGREGATION == "mean":
    agg = fi.groupby(group_cols, dropna=False)["Importance"].agg(["mean", "std"])
    agg = agg.rename(columns={"mean": "Importance", "std": "Importance_std"})
else:
    raise ValueError("AGGREGATION must be 'mean' or 'sum'.")

meta = (
    fi.groupby(group_cols, dropna=False)[["Type", "Source"]]
      .agg(first_non_null)
)

overall = agg.reset_index().merge(meta.reset_index(), on=group_cols, how="left")

overall = overall.sort_values(["Model", "Importance"], ascending=[True, False])

overall = overall[["Model", "Feature", "Importance", "Importance_std", "Type", "Source"]]
overall.to_csv(FI_OUT, index=False)
print(f"Saved overall feature importance to: {FI_OUT}")

