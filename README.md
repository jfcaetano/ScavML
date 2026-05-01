# ScavML
Online Repository for the article: "Assessing the Radical Scavenging Potential _in Silico_: a Machine Learning and Quantum Mechanics Combined Approach"

<div align="center">
  <img src="Supporting Information/scaveML-SI.png" alt="image" width="550" height="300">
</div>

## Script Overview
_rdkit-calc.py_: This script takes a CSV dataset of molecules, identifies all medium–radical combinations, and for each pair computes a series of RDKit descriptors for both the base SMILES and the radical SMILES. It assembles these descriptors into a new, expanded table as a CSV file.

_model-calc.py_: This script performs a leave-one-molecule-out regression study on the descriptor dataset using several machine learning models, computing performance metrics, feature importances, and per-sample prediction errors. It adds structural and descriptor-category annotations, summarizing each model’s class performance with uncertainty estimates.

_validation.py_: The code carries out a parallel grid search to tune a Random Forest regressor on the descriptor dataset, using a train–test split with preprocessing for categorical and numeric features and records each hyperparameter set’s test R² and MAE in a CSV file.

_feat-importance.py_: Script augments the per-molecule feature-importance table with SMILES class labels and descriptor metadata, then aggregates and normalizes feature importances by model and SMILES class, and finally builds an overall feature-importance summary with standard deviations.

_learning-curve-opt1.py_: Script generates learning curves for the descriptor-based regression workflow, tracking model performance across increasing training-set sizes to evaluate data generalization behavior.

_lomo-categorical-only.py_: This script carries out a leave-one-molecule-out regression study using only categorical descriptors as a baseline for the pipeline.

_lomo-random-split-control.py_: Script implements a random-split control analysis for the leave-one-molecule-out workflow, training and testing models on randomly partitioned data.
