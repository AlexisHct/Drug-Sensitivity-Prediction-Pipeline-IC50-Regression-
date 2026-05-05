"""
Project: Drug Sensitivity Prediction Pipeline
Copyright (C) 2026 Alexis Hucteau

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def prepare_results(prediction, metadata):

    metadata['Predicted'] = prediction

    return metadata

def top_lignees(results_df, nom_analyse):

    sorted_df = results_df.sort_values(by=['CELL_LINE_NAME'])

    print("Les 10 lignées cellulaire les plus sensibles :")
    print(sorted_df.tail(10))

    plt.figure(figsize=(24,6))
    sns.boxplot(data=sorted_df, x='CELL_LINE_NAME', y='Predicted', palette='vlag')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution du pIC50 prédit en fonction des lignées cellulaire')
    plt.ylabel('pIC50 prédit')
    plt.tight_layout()
    plt.savefig(f'projects/{nom_analyse}/IC50_par_lignee.png')
    plt.show()

    return sorted_df

def top_cancer(results_df, nom_analyse):

    sorted_df = results_df.sort_values(by=['CANCER_TYPE'])

    print("Les 10 types de cancer les plus sensibles :")
    print(sorted_df.tail(10))

    plt.figure(figsize=(24,6))
    sns.boxplot(data=sorted_df, x='CANCER_TYPE', y='Predicted', palette='vlag')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution du pIC50 prédit en fonction des types de cancer')
    plt.ylabel('pIC50 prédit')
    plt.tight_layout()
    plt.savefig(f'projects/{nom_analyse}/IC50_par_cancer.png')
    plt.show()
