#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:41:37 2026

@author: jfcaetano

Learning curve by number of training molecules for Opt1.

"""

import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


DATA_FILE = "Davide-RDKIT.csv"
Y_COLUMN = "ENERGY"

categorical_columns = [
    'MEDIUM', 'SITE', 'RADICAL', 'GROUP',
    'SMILES_Superclass', 'SMILES_Class', 'SMILES_Parent_Level'
]

non_model_cols = ['ID', 'MOLECULE', 'SMILES', 'SMILES RADICAL', 'ENERGY']

TRAIN_MOLECULE_COUNTS = [20, 40, 60, 70]
N_REPEATS = 30
RANDOM_SEED = 47
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

MODEL_NAME = "RandomForestOpt1"
BASE_MODEL = RandomForestRegressor(
    n_estimators=300,
    n_jobs=-1,
    random_state=47)

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    residual_sd = np.std(y_true - y_pred, ddof=1) if len(y_true) > 1 else 0.0
    return r2, mae, rmse, residual_sd


def bootstrap_metric_sds(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if n < 2:
        return np.nan, np.nan, np.nan

    r2_vals, mae_vals, rmse_vals = [], [], []

    for _ in tqdm(range(n_boot), desc="Bootstrapping summary metrics", leave=False):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]

        r2_vals.append(r2_score(yt, yp) if len(yt) > 1 else np.nan)
        mae_vals.append(mean_absolute_error(yt, yp))
        rmse_vals.append(np.sqrt(mean_squared_error(yt, yp)))

    return (
        np.nanstd(r2_vals, ddof=1),
        np.nanstd(mae_vals, ddof=1),
        np.nanstd(rmse_vals, ddof=1)
    )


def build_features(train_df, test_df, numeric_columns, categorical_columns):
    X_num_train = train_df[numeric_columns].values
    X_num_test = test_df[numeric_columns].values

    enc = OneHotEncoder(
        drop='first',
        handle_unknown='ignore',
        sparse_output=False
    )

    X_cat_train = enc.fit_transform(train_df[categorical_columns])
    X_cat_test = enc.transform(test_df[categorical_columns])

    X_train = np.hstack([X_num_train, X_cat_train])
    X_test = np.hstack([X_num_test, X_cat_test])

    return X_train, X_test


print("Loading data...")
df = pd.read_csv(DATA_FILE)

numeric_columns = [
    c for c in df.columns
    if c not in non_model_cols + categorical_columns
]

required_cols = set(['ID', 'MOLECULE', Y_COLUMN] + categorical_columns + numeric_columns)
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in {DATA_FILE}: {missing_cols}")

df = df.copy().reset_index(drop=True)

all_molecules = np.array(sorted(df["MOLECULE"].astype(str).unique()))
n_total_molecules = len(all_molecules)

print(f"Total unique molecules: {n_total_molecules}")

for n_train in TRAIN_MOLECULE_COUNTS:
    if n_train >= n_total_molecules:
        raise ValueError(
            f"Training molecule count {n_train} must be smaller than total molecules {n_total_molecules}."
        )

# Learning curve 
rng = np.random.default_rng(RANDOM_SEED)

per_repeat_rows = []
all_prediction_rows = []

outer_total = len(TRAIN_MOLECULE_COUNTS) * N_REPEATS
progress = tqdm(total=outer_total, desc="Learning-curve runs", unit="run")

for n_train in TRAIN_MOLECULE_COUNTS:
    for repeat in range(1, N_REPEATS + 1):

        train_molecules = rng.choice(all_molecules, size=n_train, replace=False)
        train_molecules = np.array(sorted(train_molecules))

        test_molecules = np.array(sorted(np.setdiff1d(all_molecules, train_molecules)))

        train_df = df[df["MOLECULE"].astype(str).isin(train_molecules)].copy()
        test_df = df[df["MOLECULE"].astype(str).isin(test_molecules)].copy()

        X_train, X_test = build_features(
            train_df=train_df,
            test_df=test_df,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns
        )

        y_train = train_df[Y_COLUMN].values
        y_test = test_df[Y_COLUMN].values

        model = clone(BASE_MODEL)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        r2_train, mae_train, rmse_train, resid_sd_train = compute_metrics(y_train, y_train_pred)
        r2_test, mae_test, rmse_test, resid_sd_test = compute_metrics(y_test, y_test_pred)

        per_repeat_rows.append({
            "Analysis": "MoleculeCountLearningCurve",
            "Model": MODEL_NAME,
            "Train_Molecule_Count": n_train,
            "Repeat": repeat,
            "Train_Row_Count": len(train_df),
            "Test_Row_Count": len(test_df),
            "Test_Molecule_Count": len(test_molecules),
            "R2_Train": r2_train,
            "MAE_Train": mae_train,
            "RMSE_Train": rmse_train,
            "Residuals_SD_Train": resid_sd_train,
            "R2_Test": r2_test,
            "MAE_Test": mae_test,
            "RMSE_Test": rmse_test,
            "Residuals_SD_Test": resid_sd_test
        })

        pred_df = pd.DataFrame({
            "Analysis": "MoleculeCountLearningCurve",
            "Model": MODEL_NAME,
            "Train_Molecule_Count": n_train,
            "Repeat": repeat,
            "ID": test_df["ID"].values,
            "MOLECULE": test_df["MOLECULE"].values,
            "SITE": test_df["SITE"].values if "SITE" in test_df.columns else np.nan,
            "MEDIUM": test_df["MEDIUM"].values if "MEDIUM" in test_df.columns else np.nan,
            "RADICAL": test_df["RADICAL"].values if "RADICAL" in test_df.columns else np.nan,
            "SMILES_Class": test_df["SMILES_Class"].values if "SMILES_Class" in test_df.columns else np.nan,
            "True Energy": y_test,
            "Predicted Energy": y_test_pred,
            "Absolute Error": np.abs(y_test - y_test_pred)
        })

        all_prediction_rows.append(pred_df)
        progress.update(1)

progress.close()


per_repeat_df = pd.DataFrame(per_repeat_rows)
all_predictions_df = pd.concat(all_prediction_rows, ignore_index=True)

per_repeat_df.to_csv("opt1_learning_curve_per_repeat.csv", index=False)


# Aggregated summary
summary_rows = []

for n_train in tqdm(TRAIN_MOLECULE_COUNTS, desc="Summarizing learning curve", unit="size"):
    subset_metrics = per_repeat_df[per_repeat_df["Train_Molecule_Count"] == n_train].copy()
    subset_preds = all_predictions_df[all_predictions_df["Train_Molecule_Count"] == n_train].copy()

    y_true = subset_preds["True Energy"].values
    y_pred = subset_preds["Predicted Energy"].values

    pooled_r2, pooled_mae, pooled_rmse, pooled_resid_sd = compute_metrics(y_true, y_pred)
    sd_r2_boot, sd_mae_boot, sd_rmse_boot = bootstrap_metric_sds(
        y_true, y_pred,
        n_boot=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED + int(n_train)
    )

    summary_rows.append({
        "Analysis": "MoleculeCountLearningCurve",
        "Model": MODEL_NAME,
        "Train_Molecule_Count": n_train,
        "N_Repeats": N_REPEATS,
        "Mean_Train_Row_Count": subset_metrics["Train_Row_Count"].mean(),
        "Mean_Test_Row_Count": subset_metrics["Test_Row_Count"].mean(),
        "Mean_Test_Molecule_Count": subset_metrics["Test_Molecule_Count"].mean(),

        "Mean_R2_Test": subset_metrics["R2_Test"].mean(),
        "SD_R2_Test": subset_metrics["R2_Test"].std(ddof=1),
        "Mean_MAE_Test": subset_metrics["MAE_Test"].mean(),
        "SD_MAE_Test": subset_metrics["MAE_Test"].std(ddof=1),
        "Mean_RMSE_Test": subset_metrics["RMSE_Test"].mean(),
        "SD_RMSE_Test": subset_metrics["RMSE_Test"].std(ddof=1),

        "Pooled_R2": pooled_r2,
        "Pooled_R2_bootstrap_SD": sd_r2_boot,
        "Pooled_MAE": pooled_mae,
        "Pooled_MAE_bootstrap_SD": sd_mae_boot,
        "Pooled_RMSE": pooled_rmse,
        "Pooled_RMSE_bootstrap_SD": sd_rmse_boot,
        "Pooled_Residuals_SD": pooled_resid_sd
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("opt1_learning_curve_summary.csv", index=False)
print("End")
