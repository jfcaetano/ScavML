#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:32:14 2026

@author: jfcaetano

Unified Opt1 validation script:
1) Leave-one-molecule-out
2) Random row split control

"""

import math
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
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

# Random split settings
N_RANDOM_SPLITS = 50
RANDOM_TEST_SIZE = 0.20
RANDOM_STATE = 47

# Bootstrap settings for aggregated uncertainty
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

# Opt1 model only
MODEL_NAME = "RandomForestOpt1"
BASE_MODEL = RandomForestRegressor(
    n_estimators=300,
    n_jobs=-1,
    random_state=47
)



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

    for _ in tqdm(range(n_boot), desc="Bootstrapping pooled metrics", leave=False):
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


def summarize_predictions(pred_df, analysis_name, n_units, unit_label):
    y_true = pred_df["True Energy"].values
    y_pred = pred_df["Predicted Energy"].values

    pooled_r2, pooled_mae, pooled_rmse, pooled_resid_sd = compute_metrics(y_true, y_pred)
    sd_r2, sd_mae, sd_rmse = bootstrap_metric_sds(
        y_true, y_pred, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )

    row = {
        "Analysis": analysis_name,
        "Model": MODEL_NAME,
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



print("Loading data...")
data = pd.read_csv(DATA_FILE)

numeric_columns = [
    c for c in data.columns
    if c not in non_model_cols + categorical_columns
]

required_cols = set(['ID', 'MOLECULE', Y_COLUMN] + categorical_columns + numeric_columns)
missing_cols = required_cols - set(data.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in {DATA_FILE}: {missing_cols}")

df = data.copy().reset_index(drop=True)


# 1) LOMO analysis
print("Running LOMO analysis...")
unique_molecules = np.unique(df["MOLECULE"].values)

lomo_fold_rows = []
lomo_pred_rows = []

for molecule in tqdm(unique_molecules, desc="LOMO folds", unit="molecule"):
    idx_train = df["MOLECULE"].values != molecule
    idx_test = df["MOLECULE"].values == molecule

    train_df = df.loc[idx_train].copy()
    test_df = df.loc[idx_test].copy()

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
        "Residuals_SD_Test": resid_sd_test
    })

    fold_pred_df = pd.DataFrame({
        "Analysis": "LOMO",
        "Fold_or_Split": molecule,
        "Model": MODEL_NAME,
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

    lomo_pred_rows.append(fold_pred_df)

lomo_fold_metrics_df = pd.DataFrame(lomo_fold_rows)
lomo_predictions_df = pd.concat(lomo_pred_rows, ignore_index=True)

lomo_fold_metrics_df.to_csv("opt1_lomo_fold_metrics.csv", index=False)
lomo_predictions_df.to_csv("opt1_lomo_predictions.csv", index=False)


# 2) Random row split analysis
print("Running random row split analysis...")
random_splitter = ShuffleSplit(
    n_splits=N_RANDOM_SPLITS,
    test_size=RANDOM_TEST_SIZE,
    random_state=RANDOM_STATE
)

random_split_rows = []
random_pred_rows = []

split_indices = list(random_splitter.split(df))

for split_idx, (train_idx, test_idx) in tqdm(
    enumerate(split_indices, start=1),
    total=len(split_indices),
    desc="Random splits",
    unit="split"
):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

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

    train_molecules = set(train_df["MOLECULE"].astype(str))
    test_molecules = set(test_df["MOLECULE"].astype(str))
    overlap_count = len(train_molecules.intersection(test_molecules))

    random_split_rows.append({
        "Fold": split_idx,
        "Analysis": "RandomRowSplit",
        "Model": MODEL_NAME,
        "Train_n": len(train_df),
        "Test_n": len(test_df),
        "Unique_Molecules_Train": len(train_molecules),
        "Unique_Molecules_Test": len(test_molecules),
        "Overlapping_Molecules_Train_Test": overlap_count,
        "R2_Train": r2_train,
        "MAE_Train": mae_train,
        "RMSE_Train": rmse_train,
        "Residuals_SD_Train": resid_sd_train,
        "R2_Test": r2_test,
        "MAE_Test": mae_test,
        "RMSE_Test": rmse_test,
        "Residuals_SD_Test": resid_sd_test
    })

    split_pred_df = pd.DataFrame({
        "Analysis": "RandomRowSplit",
        "Fold_or_Split": split_idx,
        "Model": MODEL_NAME,
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

    random_pred_rows.append(split_pred_df)

random_split_metrics_df = pd.DataFrame(random_split_rows)
random_predictions_df = pd.concat(random_pred_rows, ignore_index=True)

random_split_metrics_df.to_csv("opt1_random_split_metrics.csv", index=False)
random_predictions_df.to_csv("opt1_random_predictions.csv", index=False)


# 3) Aggregated comparison
print("Building aggregated results...")
aggregated_rows = []

aggregated_rows.append(
    summarize_predictions(
        pred_df=lomo_predictions_df,
        analysis_name="LOMO",
        n_units=len(unique_molecules),
        unit_label="molecules"
    )
)

aggregated_rows[-1]["Mean_R2_across_units"] = lomo_fold_metrics_df["R2_Test"].mean()
aggregated_rows[-1]["SD_R2_across_units"] = lomo_fold_metrics_df["R2_Test"].std(ddof=1)
aggregated_rows[-1]["Mean_MAE_across_units"] = lomo_fold_metrics_df["MAE_Test"].mean()
aggregated_rows[-1]["SD_MAE_across_units"] = lomo_fold_metrics_df["MAE_Test"].std(ddof=1)
aggregated_rows[-1]["Mean_RMSE_across_units"] = lomo_fold_metrics_df["RMSE_Test"].mean()
aggregated_rows[-1]["SD_RMSE_across_units"] = lomo_fold_metrics_df["RMSE_Test"].std(ddof=1)
aggregated_rows[-1]["Mean_Overlap_Train_Test_Molecules"] = 0.0

aggregated_rows.append(
    summarize_predictions(
        pred_df=random_predictions_df,
        analysis_name="RandomRowSplit",
        n_units=N_RANDOM_SPLITS,
        unit_label="splits"
    )
)

aggregated_rows[-1]["Mean_R2_across_units"] = random_split_metrics_df["R2_Test"].mean()
aggregated_rows[-1]["SD_R2_across_units"] = random_split_metrics_df["R2_Test"].std(ddof=1)
aggregated_rows[-1]["Mean_MAE_across_units"] = random_split_metrics_df["MAE_Test"].mean()
aggregated_rows[-1]["SD_MAE_across_units"] = random_split_metrics_df["MAE_Test"].std(ddof=1)
aggregated_rows[-1]["Mean_RMSE_across_units"] = random_split_metrics_df["RMSE_Test"].mean()
aggregated_rows[-1]["SD_RMSE_across_units"] = random_split_metrics_df["RMSE_Test"].std(ddof=1)
aggregated_rows[-1]["Mean_Overlap_Train_Test_Molecules"] = random_split_metrics_df["Overlapping_Molecules_Train_Test"].mean()

aggregated_df = pd.DataFrame(aggregated_rows)

if len(aggregated_df) == 2:
    lomo_row = aggregated_df[aggregated_df["Analysis"] == "LOMO"].iloc[0]
    rand_row = aggregated_df[aggregated_df["Analysis"] == "RandomRowSplit"].iloc[0]

    diff_row = {
        "Analysis": "Difference(RandomRowSplit-LOMO)",
        "Model": MODEL_NAME,
        "N_units": np.nan,
        "Unit_type": "",
        "N_predictions": rand_row["N_predictions"] - lomo_row["N_predictions"],
        "Pooled_R2": rand_row["Pooled_R2"] - lomo_row["Pooled_R2"],
        "Pooled_R2_bootstrap_SD": np.nan,
        "Pooled_MAE": rand_row["Pooled_MAE"] - lomo_row["Pooled_MAE"],
        "Pooled_MAE_bootstrap_SD": np.nan,
        "Pooled_RMSE": rand_row["Pooled_RMSE"] - lomo_row["Pooled_RMSE"],
        "Pooled_RMSE_bootstrap_SD": np.nan,
        "Pooled_Residuals_SD": rand_row["Pooled_Residuals_SD"] - lomo_row["Pooled_Residuals_SD"],
        "Mean_R2_across_units": rand_row["Mean_R2_across_units"] - lomo_row["Mean_R2_across_units"],
        "SD_R2_across_units": np.nan,
        "Mean_MAE_across_units": rand_row["Mean_MAE_across_units"] - lomo_row["Mean_MAE_across_units"],
        "SD_MAE_across_units": np.nan,
        "Mean_RMSE_across_units": rand_row["Mean_RMSE_across_units"] - lomo_row["Mean_RMSE_across_units"],
        "SD_RMSE_across_units": np.nan,
        "Mean_Overlap_Train_Test_Molecules": rand_row["Mean_Overlap_Train_Test_Molecules"] - lomo_row["Mean_Overlap_Train_Test_Molecules"]
    }

    aggregated_df = pd.concat([aggregated_df, pd.DataFrame([diff_row])], ignore_index=True)

aggregated_df.to_csv("opt1_validation_aggregated_results.csv", index=False)


print("~End~")
