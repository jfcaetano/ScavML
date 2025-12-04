# ScavML
Online Repository for the article: "Assessing the Radical Scavenging Potential _in Silico_: a Machine Learning and Quantum Mechanics Combined Approach"

<img width="743" height="359" alt="Screenshot 2025-12-04 at 16 36 02" src="https://github.com/user-attachments/assets/5babcfd0-6b2f-4f7b-8707-53c6b0d3274e" />

## Script Overview
_rdkit-calc.py_: This script takes a CSV dataset of molecules, identifies all medium–radical combinations, and for each pair computes a series of RDKit descriptors for both the base SMILES and the radical SMILES. It assembles these descriptors into a new, expanded table as a CSV file.

_model-calc.py_: This script performs a leave-one-molecule-out regression study on the descriptor dataset using several machine learning models, computing performance metrics, feature importances, and per-sample prediction errors. It adds structural and descriptor-category annotations, summarizing each model’s class performance with uncertainty estimates.

_validation.py_: The code carries out a parallel grid search to tune a Random Forest regressor on the descriptor dataset, using a train–test split with preprocessing for categorical and numeric features and records each hyperparameter set’s test R² and MAE in a CSV file.

_feat-importance.py_: Script augments the per-molecule feature-importance table with SMILES class labels and descriptor metadata, then aggregates and normalizes feature importances by model and SMILES class, and finally builds an overall feature-importance summary with standard deviations.
