#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Apr 15 20:32:14 2026

@author: jfcaetano

Categorical-only LOMO baseline.
LOMO Random Forest model using only
one-hot encodings of:

    RADICAL
    MEDIUM
    SITE
    GROUP
    SMILES_Class
    SMILES_Superclass
    SMILES_Parent_Level
     
    
"""

import math
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Settings
DATA_FILE = "Davide-RDKIT.csv"
Y_COLUMN = "ENERGY"

MODEL_NAME = "RandomForest_CategoricalOnly_LOMO"

categorical_columns = [
    "RADICAL",
    "MEDIUM",
    "SITE",
    "GROUP",
    "SMILES_Class",
    "SMILES_Superclass",
    "SMILES_Parent_Level"]


N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

BASE_MODEL = RandomForestRegressor(
    n_estimators=300,
    n_jobs=-1,
    random_state=47)


# Functions

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

    r2_vals = []
    mae_vals = []
    rmse_vals = []

    for _ in tqdm(
        range(n_boot),
        desc="Bootstrapping pooled metrics",
        leave=False
    ):
        idx = rng.integers(0, n, n)

        yt = y_true[idx]
        yp = y_pred[idx]

        r2_vals.append(r2_score(yt, yp) if len(yt) > 1 else np.nan)
        mae_vals.append(mean_absolute_error(yt, yp))
        rmse_vals.append(np.sqrt(mean_squared_error(yt, yp)))

    sd_r2 = np.nanstd(r2_vals, ddof=1)
    sd_mae = np.nanstd(mae_vals, ddof=1)
    sd_rmse = np.nanstd(rmse_vals, ddof=1)

    return sd_r2, sd_mae, sd_rmse


def build_categorical_features(train_df, test_df, categorical_columns):
    enc = OneHotEncoder(
        drop=None,
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train = enc.fit_transform(train_df[categorical_columns].astype(str))
    X_test = enc.transform(test_df[categorical_columns].astype(str))

    return X_train, X_test


def summarize_predictions(pred_df, analysis_name, model_name, n_units, unit_label):
    y_true = pred_df["True Energy"].values
    y_pred = pred_df["Predicted Energy"].values

    pooled_r2, pooled_mae, pooled_rmse, pooled_resid_sd = compute_metrics(
        y_true,
        y_pred
    )

    sd_r2, sd_mae, sd_rmse = bootstrap_metric_sds(
        y_true,
        y_pred,
        n_boot=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED
    )

    row = {
        "Analysis": analysis_name,
        "Model": model_name,
        "N_units": n_units,
        "Unit_type": unit_label,
        "N_predictions": len(pred_df),
        "Pooled_R2": pooled_r2,
        "Pooled_R2_bootstrap_SD": sd_r2,
        "Pooled_MAE": pooled_mae,
        "Pooled_MAE_bootstrap_SD": sd_mae,
        "Pooled_RMSE": pooled_rmse,
        "Pooled_RMSE_bootstrap_SD": sd_rmse,
        "Pooled_Residuals_SD": pooled_resid_sd
    }

    return row


# Load data
print("Loading data...")

df = pd.read_csv(DATA_FILE)
df = df.copy().reset_index(drop=True)

required_cols = set(["ID", "MOLECULE", Y_COLUMN] + categorical_columns)
missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(
        f"Missing required columns in {DATA_FILE}: {missing_cols}")


# LOMO categorical-only baseline

print("Running categorical-only LOMO baseline...")

unique_molecules = np.unique(df["MOLECULE"].values)

lomo_fold_rows = []
lomo_pred_rows = []

for molecule in tqdm(unique_molecules, desc="LOMO folds", unit="molecule"):
    idx_train = df["MOLECULE"].values != molecule
    idx_test = df["MOLECULE"].values == molecule

    train_df = df.loc[idx_train].copy()
    test_df = df.loc[idx_test].copy()

    X_train, X_test = build_categorical_features(
        train_df=train_df,
        test_df=test_df,
        categorical_columns=categorical_columns)

    y_train = train_df[Y_COLUMN].values
    y_test = test_df[Y_COLUMN].values

    model = clone(BASE_MODEL)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train, mae_train, rmse_train, resid_sd_train = compute_metrics(
        y_train,
        y_train_pred)

    r2_test, mae_test, rmse_test, resid_sd_test = compute_metrics(
        y_test,
        y_test_pred)

    lomo_fold_rows.append({
        "Fold": molecule,
        "Analysis": "LOMO",
        "Model": MODEL_NAME,
        "Train_n": len(train_df),
        "Test_n": len(test_df),
        "R2_Train": r2_train,
        "MAE_Train": mae_train,
        "RMSE_Train": rmse_train,
        "Residuals_SD_Train": resid_sd_train,
        "R2_Test": r2_test,
        "MAE_Test": mae_test,
        "RMSE_Test": rmse_test,
        "Residuals_SD_Test": resid_sd_test})

    fold_pred_df = pd.DataFrame({
        "Analysis": "LOMO",
        "Fold_or_Split": molecule,
        "Model": MODEL_NAME,
        "ID": test_df["ID"].values,
        "MOLECULE": test_df["MOLECULE"].values,
        "SITE": test_df["SITE"].values,
        "MEDIUM": test_df["MEDIUM"].values,
        "RADICAL": test_df["RADICAL"].values,
        "SMILES_Class": test_df["SMILES_Class"].values,
        "True Energy": y_test,
        "Predicted Energy": y_test_pred,
        "Absolute Error": np.abs(y_test - y_test_pred)})

    lomo_pred_rows.append(fold_pred_df)


# Save fold and prediction-level results

lomo_fold_metrics_df = pd.DataFrame(lomo_fold_rows)
lomo_predictions_df = pd.concat(lomo_pred_rows, ignore_index=True)

lomo_fold_metrics_df.to_csv(
    "baseline_categorical_only_lomo_fold_metrics.csv",
    index=False)

lomo_predictions_df.to_csv(
    "baseline_categorical_only_lomo_predictions.csv",
    index=False)


# Save aggregated results
print("Building aggregated results...")

aggregated_row = summarize_predictions(
    pred_df=lomo_predictions_df,
    analysis_name="LOMO",
    model_name=MODEL_NAME,
    n_units=len(unique_molecules),
    unit_label="molecules")

aggregated_row["Mean_R2_across_units"] = lomo_fold_metrics_df["R2_Test"].mean()
aggregated_row["SD_R2_across_units"] = lomo_fold_metrics_df["R2_Test"].std(ddof=1)

aggregated_row["Mean_MAE_across_units"] = lomo_fold_metrics_df["MAE_Test"].mean()
aggregated_row["SD_MAE_across_units"] = lomo_fold_metrics_df["MAE_Test"].std(ddof=1)

aggregated_row["Mean_RMSE_across_units"] = lomo_fold_metrics_df["RMSE_Test"].mean()
aggregated_row["SD_RMSE_across_units"] = lomo_fold_metrics_df["RMSE_Test"].std(ddof=1)

aggregated_row["Mean_Overlap_Train_Test_Molecules"] = 0.0

aggregated_df = pd.DataFrame([aggregated_row])

aggregated_df.to_csv(
    "baseline_categorical_only_lomo_aggregated_results.csv",
    index=False)


print("Saved files:")
print("baseline_categorical_only_lomo_fold_metrics.csv")
print("baseline_categorical_only_lomo_predictions.csv")
print("baseline_categorical_only_lomo_aggregated_results.csv")


print("~End~")
