#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 11:35:17 2025

@author: jfcaetano
"""


import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors

def calculate_descriptor(desc, mol):
    if desc == "BalabanJ":
        return GraphDescriptors.BalabanJ(mol)
    else:
        func = getattr(Descriptors, desc)
        return func(mol)

def transform_csv(input_file, output_file):
    df = pd.read_csv(input_file, delimiter=',')
    

    radical_types = set()
    medium_types = set()
    

    for col in df.columns:
        if '-' in col and not col.startswith('SMILES'):
            medium, radical = col.split('-')
            medium_types.add(medium)
            radical_types.add(radical)
    
    # Define descriptors to calculate
    my_descriptors = [desc_name for desc_name in dir(Descriptors) if desc_name in ['BalabanJ','BertzCT','TPSA'] or desc_name[:3] == 'Chi' or 'VSA' in desc_name or 'Kappa' in desc_name or desc_name[:1] in ['H', 'N', 'M']]
    
    transformed_rows = []
    row_id = 1
    for _, row in df.iterrows():
        for medium in medium_types:
            for radical in radical_types:
                energy_col = f'{medium}-{radical}'
                smiles_col = f'SMILES-{radical}'
                
                if energy_col in df.columns and smiles_col in df.columns:
                    energy = row[energy_col]
                    radical_smiles = row[smiles_col]
                    
                    # Calculate descriptors for SMILES and SMILES RADICAL
                    smiles_mol = Chem.MolFromSmiles(row['SMILES'])
                    radical_mol = Chem.MolFromSmiles(radical_smiles) if radical_smiles else None
                    
                    smiles_descriptors = {f'SMILES_{desc}': calculate_descriptor(desc, smiles_mol) for desc in my_descriptors} if smiles_mol else {}
                    radical_descriptors = {f'SMILES_RADICAL_{desc}': calculate_descriptor(desc, radical_mol) for desc in my_descriptors} if radical_mol else {}
                    
                    transformed_rows.append([
                        row_id, row['MOLECULE'], row['SMILES'], row['SITE'], row['GROUP'], row['CHARGE'], row['SMILES_Superclass'], row['SMILES_Class'], row['SMILES_Parent_Level'],
                                                
                        medium, radical, energy, radical_smiles,
                        *smiles_descriptors.values(), *radical_descriptors.values()
                    ])
                    row_id += 1
    

    descriptor_columns = [f'SMILES_{desc}' for desc in my_descriptors] + [f'SMILES_RADICAL_{desc}' for desc in my_descriptors]
    transformed_df = pd.DataFrame(transformed_rows, columns=[
        'ID', 'MOLECULE', 'SMILES', 'SITE', 'GROUP', 'CHARGE', 'SMILES_Superclass', 'SMILES_Class', 'SMILES_Parent_Level', 'MEDIUM', 'RADICAL', 'ENERGY', 'SMILES RADICAL',
        *descriptor_columns
    ])
    

    transformed_df.to_csv(output_file, index=False, sep=',')
    
    return transformed_df

#RUN
transform_csv('Davide_full_corretto.csv', 'Davide-RDKIT.csv')

