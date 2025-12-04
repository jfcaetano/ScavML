# ScavML
Online Repository for the article: "Assessing the Radical Scavenging Potential _in Silico_: a Machine Learning and Quantum Mechanics Combined Approach"

# Script Overview
## rdkit-calc.py
This script takes an input CSV of molecules, identifies all medium–radical combinations, and for each pair computes a series of RDKit molecular descriptors for both the base SMILES and the radical SMILES. It then assembles these descriptors into a new, expanded table and exports it as a CSV file.
