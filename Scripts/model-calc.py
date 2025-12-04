#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:01:22 2025

@author: jfcaetano
"""

import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib 
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression


warnings.filterwarnings("ignore")

data = pd.read_csv("Davide-RDKIT.csv")
categorical_columns = ['MEDIUM', 'SITE', 'RADICAL', 'GROUP',
                       'SMILES_Superclass', 'SMILES_Class', 'SMILES_Parent_Level']

Y_column = 'ENERGY'

non_model_cols = ['ID', 'MOLECULE', 'SMILES', 'SMILES RADICAL', 'ENERGY']
numeric_columns = [c for c in data.columns
                   if c not in non_model_cols + categorical_columns]

X_num_all = data[numeric_columns]
X_cat_all = data[categorical_columns]
Y_all = data[Y_column].values
molecule_labels = data["MOLECULE"].values
ids = data["ID"].values


# Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=47),
    "GradientBoosting": GradientBoostingRegressor(random_state=47),
    "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=100, random_state=47),
    "RandomForestOpt1": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=47)}

stats_results = []
feature_importance_results = []
predictions_results = []

unique_molecules = np.unique(molecule_labels)



def process_molecule(molecule):
    idx_train = molecule_labels != molecule
    idx_test  = molecule_labels == molecule

    X_num_train = X_num_all.iloc[idx_train].values
    X_num_test  = X_num_all.iloc[idx_test].values

    enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    X_cat_train = enc.fit_transform(X_cat_all.iloc[idx_train])
    X_cat_test  = enc.transform(X_cat_all.iloc[idx_test])

    X_train = np.hstack([X_num_train, X_cat_train])
    X_test  = np.hstack([X_num_test,  X_cat_test])

    y_train = Y_all[idx_train]
    y_test  = Y_all[idx_test]
    test_IDs = ids[idx_test]

    enc_cat_names = enc.get_feature_names_out(categorical_columns).tolist()
    fold_feature_names = numeric_columns + enc_cat_names

    results = []
    predictions = []
    feature_importance = []

    # Train/evaluate all models defined in `models` dict
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        rsq_train = r2_score(y_train, y_train_pred)
        rsq_test  = r2_score(y_test,  y_test_pred)
        RMSE = math.sqrt(mean_squared_error(y_test, y_test_pred))
        MAE  = mean_absolute_error(y_test, y_test_pred)
        STD  = np.std(y_test_pred)

        results.append({
            "MOLECULE": molecule,
            "Model": model_name,
            "rsq_train": rsq_train,
            "rsq_test": rsq_test,
            "RMSE": RMSE,
            "MAE": MAE,
            "STD": STD
        })

        # Per-sample predictions
        for sample_id, true_val, pred_val in zip(test_IDs, y_test, y_test_pred):
            predictions.append({
                "MOLECULE": molecule,
                "ID": sample_id,
                "Model": model_name,
                "True Energy": true_val,
                "Predicted Energy": pred_val,
                "Absolute Error": abs(true_val - pred_val)
            })

        # Feature importances (tree models, etc.)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            for feat, imp in zip(fold_feature_names, importances):
                feature_importance.append({
                    "MOLECULE": molecule,
                    "Model": model_name,
                    "Feature": feat,
                    "Importance": float(imp)
                })

    return results, predictions, feature_importance


# Use joblib to parallelize
parallel_results = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(process_molecule)(mol) for mol in tqdm(unique_molecules, desc="Processing Molecules", unit="mol")
)


for res, pred, imp in parallel_results:
    stats_results.extend(res)
    predictions_results.extend(pred)
    feature_importance_results.extend(imp)

# Convert to DataFrames
stats_results_df = pd.DataFrame(stats_results)
feature_importance_df = pd.DataFrame(feature_importance_results)
predictions_results_df = pd.DataFrame(predictions_results)

# Save results
stats_results_df.to_csv("statistical_results-full-opt.csv", index=False)
feature_importance_df.to_csv("feature_importance-full-opt.csv", index=False)
predictions_results_df.to_csv("predictions_results-full-opt.csv", index=False)

print("Leave-one-molecule-out statistical analysis, feature importance, and individual predictions completed.")


# Define file paths
file1 = "predictions_results-full-opt.csv"
file2 = "Davide-RDKIT.csv"

# Load CSV files
df1 = pd.read_csv(file1)  # Main dataset
df2 = pd.read_csv(file2)  # Contains SITE, MEDIUM, and RADICAL
df2 = df2[['ID', 'MOLECULE', 'SITE', 'MEDIUM', 'RADICAL', 'SMILES_Class']]

df_merged = df1.merge(df2, on=['ID', 'MOLECULE'], how='left')


output_file = "predictions_results-full-sites-opt.csv"
df_merged.to_csv(output_file, index=False)


print(f"Merging completed. The updated file is saved as {output_file}.")

### Descriptor Groups labelling 

file1 = "feature_importance-full-opt.csv"
file2 = "Desc_cat.csv"

df1 = pd.read_csv(file1)  
df2 = pd.read_csv(file2)  
df2 = df2[['Feature', 'Type', 'Source']]


df_merged = df1.merge(df2, on=['Feature'], how='left')

output_file = "feature_importance-full-type-opt.csv"
df_merged.to_csv(output_file, index=False)


print(f"Merging completed. The updated file is saved as {output_file}.")




# ----------------------------
# Summarize model performance with uncertanty estimations
# ----------------------------


INPUT_CSV  = "predictions_results-full-sites-opt.csv"
OUTPUT_CSV = "model_performance_with_uncertainty-opt.csv"
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(seed=42)  # reproducibility



def compute_point_metrics(y_true: pd.Series, y_pred: pd.Series):
    """
    Point estimates for R2 (nan if <2 samples), MAE, RMSE, and
    the sample standard deviation of residuals (ddof=1).
    """
    residuals = y_true - y_pred
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    resid_sd = residuals.std(ddof=1) if len(residuals) > 1 else 0.0
    return r2, mae, rmse, resid_sd

def bootstrap_sds(y_true: pd.Series, y_pred: pd.Series, n_boot: int = N_BOOTSTRAP):
    """
    Bootstrap standard deviations for R2, MAE, RMSE.
    Returns (sd_R2, sd_MAE, sd_RMSE). If fewer than 2 samples, returns NaNs.
    """
    n = len(y_true)
    if n < 2:
        return np.nan, np.nan, np.nan

    r2_vals, mae_vals, rmse_vals = [], [], []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n) 
        yt = y_true.iloc[idx]
        yp = y_pred.iloc[idx]
        r2_vals.append(r2_score(yt, yp) if len(yt) > 1 else np.nan)
        mae_vals.append(mean_absolute_error(yt, yp))
        rmse_vals.append(np.sqrt(mean_squared_error(yt, yp)))

    sd_r2   = np.nanstd(r2_vals,  ddof=1)
    sd_mae  = np.nanstd(mae_vals, ddof=1)
    sd_rmse = np.nanstd(rmse_vals,ddof=1)
    return sd_r2, sd_mae, sd_rmse


# Load data
df = pd.read_csv(INPUT_CSV)

required = {"Model", "True Energy", "Predicted Energy", "SMILES_Class"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")


rows = []

for model_name, model_data in df.groupby("Model"):
    y_true_all = model_data["True Energy"]
    y_pred_all = model_data["Predicted Energy"]

    r2, mae, rmse, resid_sd = compute_point_metrics(y_true_all, y_pred_all)
    sd_r2, sd_mae, sd_rmse = bootstrap_sds(y_true_all, y_pred_all, N_BOOTSTRAP)
    rows.append([
        model_name, "Overall",
        r2, sd_r2,
        mae, sd_mae,
        rmse, sd_rmse,
        resid_sd
    ])

    # By SMILES_Class
    for smiles_class, class_data in model_data.groupby("SMILES_Class"):
        y_true = class_data["True Energy"]
        y_pred = class_data["Predicted Energy"]

        r2, mae, rmse, resid_sd = compute_point_metrics(y_true, y_pred)
        sd_r2, sd_mae, sd_rmse = bootstrap_sds(y_true, y_pred, N_BOOTSTRAP)

        rows.append([
            model_name, smiles_class,
            r2, sd_r2,
            mae, sd_mae,
            rmse, sd_rmse,
            resid_sd
        ])


# Save
results_df = pd.DataFrame(
    rows,
    columns=[
        "Model", "SMILES_Class",
        "R2",   "R2_bootstrap_STD",
        "MAE",  "MAE_bootstrap_STD",
        "RMSE", "RMSE_bootstrap_STD",
        "Residuals_STD"
    ]
)

results_df.to_csv(OUTPUT_CSV, index=False)
print(results_df)
