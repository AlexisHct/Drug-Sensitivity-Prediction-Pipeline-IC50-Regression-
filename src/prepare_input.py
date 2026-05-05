"""
Project: Drug Sensitivity Prediction Pipeline
Copyright (C) 2026 Alexis Hucteau

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
import os
import numpy as np

def get_smiles(drug_name):
    if pd.isna(drug_name) or drug_name == "":
        return None
    
    try:
        compounds = pcp.get_compounds(drug_name, 'name')

        if compounds:
            return compounds[0].connectivity_smiles
        else:
            return None
        
    except Exception as e:
        return None

def generate_features(query):
    RDLogger.DisableLog('rdApp.*')
    morgan_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    # molecule = Chem.MolFromSmiles(query)

    # if molecule is None:
    smiles = get_smiles(query)
    molecule = Chem.MolFromSmiles(smiles)

    fp = morgan_gen.GetFingerprint(molecule)
    fps_array =np.array(fp).reshape(1,-1)
    
    fp_columns = [f"bit_{i}" for i in range (2048)]
    df_fps = pd.DataFrame(fps_array, columns = fp_columns)

    df_fps['smiles'] = smiles
    df_fps['MW'] = Descriptors.MolWt(molecule)
    df_fps['LogP'] = Descriptors.MolLogP(molecule)
    df_fps['H_Donors'] = Descriptors.NumHDonors(molecule)
    df_fps['H_Acceptors'] = Descriptors.NumHAcceptors(molecule)
    df_fps['TPSA'] = Descriptors.TPSA(molecule)
    df_fps['Rotatable_Bonds'] = Descriptors.NumRotatableBonds(molecule)

    return df_fps

def prepare_pca(pca_data, metadata, selected_cancers=None):
    
    filter_cancer = False
    
    if not selected_cancers:
        return pca_data, False

    subsets = []

    for cancer in selected_cancers:
        set = pca_data[pca_data["CANCER_TYPE"].str.contains(cancer, case=False, na=False)]
        if not set.empty:
            subsets.append(set)

    if len(subsets) > 0:
        res_df = pd.concat(subsets).drop_duplicates()
        filter_cancer = True
    else:
        res_df = pca_data
        filter_cancer = False
    
    return res_df, filter_cancer
