"""
Project: Drug Sensitivity Prediction Pipeline
Copyright (C) 2026 Alexis Hucteau

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import pandas as pd
import joblib
import os

from src.prepare_input import get_smiles, generate_features, prepare_pca
from src.inference import merge_input
from src.analysis import prepare_results, top_lignees, top_cancer

app = FastAPI(title="Drug Sensitivity Predictor")

MODEL = joblib.load('models/drug_predictor_model.pkl')
PCA_DATA = pd.read_parquet('data/processed/cell_line_pca.parquet')
METADATA = pd.read_excel('data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx', engine="openpyxl")
METADATA = METADATA[['CANCER_TYPE', 'CELL_LINE_NAME']].drop_duplicates()
PCA_DATA = pd.merge(PCA_DATA, METADATA, on='CELL_LINE_NAME', how='inner')

@app.get("/predict")
async def predict_drug_effects(
    query_name: str = Query(..., description='Nom de l\'analyse'),
    query: str = Query(..., description='Nom du médicament ou SMILES'),
    cancer_type: Optional[str] = Query(None, description = 'Liste des types de cancer d\'intérêt (optionnel)')
):
    try:
        
        query = query.strip()

        mol_features = generate_features(query)

        pca_data, filtered_cancer = prepare_pca(PCA_DATA, METADATA, cancer_type)

        X_test_data, feature_data = merge_input(mol_features, pca_data)
        
        raw_results = MODEL.predict(X_test_data)

        newpath = f'projects/{query_name}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        results_df = prepare_results(raw_results, feature_data)

        print(feature_data.columns)

        if not filtered_cancer:
            top_cancer(results_df, query_name)
        results = top_lignees(results_df, query_name)


        return {
            "status": "success",
            "query_info": {
                "name": query_name,
                "input": query,
                "cancer_filter": cancer_type
            },
            "results": {
                "results": results
            },
            "message": f"Résultats sauvegardés dans {newpath}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   