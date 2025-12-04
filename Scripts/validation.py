#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 14:41:33 2025

@author: jfcaetano
"""

import time
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


RANDOM_STATE = 47
N_JOBS_MODEL = -1   
N_JOBS_PARAM = -1
TEST_SIZE = 0.2     


df = pd.read_csv("Davide-RDKIT.csv")

y = df["ENERGY"].values

categorical = ['MEDIUM','SITE','RADICAL','GROUP',
               'SMILES_Superclass','SMILES_Class','SMILES_Parent_Level']
non_model = ['ID','MOLECULE','SMILES','SMILES RADICAL','ENERGY'] + categorical
numeric = [c for c in df.columns if c not in non_model]

X = pd.concat([df[numeric], df[categorical]], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical),
        ("num", "passthrough", numeric),
    ]
)

base = RandomForestRegressor(
    n_jobs=N_JOBS_MODEL,
    random_state=RANDOM_STATE
)

pipe = Pipeline([
    ("pre", pre),
    ("model", base),
])


param_grid = {
    "model__n_estimators":   [100, 200, 300],
    "model__max_depth":      [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 5],
    "model__max_features":   ["sqrt", "log2", None],
}
candidates = list(ParameterGrid(param_grid))
print(f"Total combinations: {len(candidates)}")

# -------------------------
# Runtime estimate
# -------------------------
def estimate_runtime(X_train, y_train, candidates, pipe):
    params0 = candidates[0]
    pipe.set_params(**params0)

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    t1 = time.perf_counter()
    fit_seconds = max(t1 - t0, 1e-6)

    total_fits = len(candidates)
    est_seconds = fit_seconds * total_fits
    est_minutes = est_seconds / 60
    est_hours = est_minutes / 60

    print("="*72)
    print("Workload forecast (before training):")
    print(f" • Candidate hyperparameter sets: {len(candidates)}")
    if est_hours >= 1:
        print(f" • Rough wall-clock estimate:     ~{est_hours:,.2f} hours")
    elif est_minutes >= 1:
        print(f" • Rough wall-clock estimate:     ~{est_minutes:,.1f} minutes")
    else:
        print(f" • Rough wall-clock estimate:     ~{est_seconds:,.1f} seconds")
    print("="*72)

estimate_runtime(X_train, y_train, candidates, pipe)


def evaluate_params(params):
    pipe.set_params(**params)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return {
        "r2_test": r2_score(y_test, y_pred),
        "mae_test": mean_absolute_error(y_test, y_pred),
        **params
    }

results = Parallel(n_jobs=N_JOBS_PARAM, verbose=10)(
    delayed(evaluate_params)(params) for params in candidates
)


results_df = pd.DataFrame(results)
results_df.to_csv("rf_holdout_r2_mae.csv", index=False)

print("Saved results to rf_holdout_r2_mae.csv")
print(results_df.head())
