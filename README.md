# ScavML
Online Repository for the article: "Assessing the Radical Scavenging Potential _in Silico_: a Machine Learning and Quantum Mechanics Combined Approach"

## Script Overview
## rdkit-calc.py
This script takes the input CSV dataset of molecules, identifies all medium–radical combinations, and for each pair computes a series of RDKit molecular descriptors for both the base SMILES and the radical SMILES. It then assembles these descriptors into a new, expanded table and exports it as a CSV file.

## model-calc.py
This script performs a leave-one-molecule-out regression study on the descriptor dataset using several machine learning models, computing performance metrics, feature importances, and per-sample prediction errors. It  enriches these outputs with structural and descriptor-category annotations, and finally summarizes each model’s overall and class-resolved performance with bootstrap-based uncertainty estimates.

## validation.py
The code carries out a parallel grid search to tune a Random Forest regressor on the descriptor dataset, using a train–test split with appropriate preprocessing for categorical and numeric features and records each hyperparameter set’s test R² and MAE in a CSV file.

## feat-importance.py
Script augments the per-molecule feature-importance table with SMILES class labels and descriptor metadata, then aggregates and normalizes feature importances by model and SMILES class, and finally builds an overall feature-importance summary with standard deviations.
