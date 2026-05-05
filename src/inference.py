"""
Project: Drug Sensitivity Prediction Pipeline
Copyright (C) 2026 Alexis Hucteau

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

def merge_input(chem_features_df, pca_data):
    chem_features_df['MERGE_ID'] = 1
    pca_data['MERGE_ID'] = 1
    
    features_data = pd.merge(chem_features_df, pca_data, on='MERGE_ID', how='inner')
    cols_to_exclude = ['TARGET', 'DRUG_ID', 'DRUG_NAME', 'CANCER_TYPE', 'CELL_LINE_NAME', 'SANGER_MODEL_ID', 'Cell_Line',
                            'ModelID', 'MERGE_ID', 'CellLineName', 'LN_IC50', 'AUC', 'RMSE', 'smiles', 'pIC50']
    final_data = features_data.drop(columns=[c for c in cols_to_exclude if c in features_data.columns])
    features_data = features_data.drop(columns=[c for c in final_data.columns if c in features_data.columns])

    return final_data, features_data
