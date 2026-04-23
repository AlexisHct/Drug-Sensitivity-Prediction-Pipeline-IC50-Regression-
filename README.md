![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)

# Drug-Sensitivity-Prediction-Pipeline-IC50-Regression-
Prédire la réponse d'une lignée cellulaire (données génomiques) à une molécule chimique donnée (données moléculaires).

## Présentation du projet

Ce projet implémente une pipeline de Machine Learning capable de prédire la réponse d'une lignée cellulaire à une molécule donnée à partir de son profil d'expression géniques.

## Architecture Technique & Choix Stratégiques

### 1. Jeux de données

* [Données de réponse aux traitements GDSC2 IC50 Data](https://cellmodelpassports.sanger.ac.uk/downloads) 
* [Données d'expression génique des lignées cellulaires](https://depmap.org/portal/data_page/?tab=currentRelease)
* [Caractéristiques chimiques des molécules](https://cellmodelpassports.sanger.ac.uk/downloads)




drug-sensitivity-pipeline/
├── data/               # Dossier ignoré par Git (.gitignore)
│   ├── raw/            # Fichiers téléchargés (GDSC, DepMap)
│   └── processed/      # Fichiers .parquet prêts pour le ML
├── notebooks/          # Travail exploratoire (Copyright Dr. X)
│   ├── 01_eda_genomics.ipynb
│   └── 02_eda_compounds.ipynb
├── src/                # Le code source "pro" (GPL v3)
│   ├── data_ingestion.py    # Script PubChemPy pour les SMILES
│   ├── preprocessing.py     # Filtrage gènes + RDKit fingerprints
│   ├── model.py             # Architecture XGBoost/PyTorch
│   └── train.py             # Script d'entraînement avec MLflow
├── models/             # Modèles sauvegardés (.pkl, .h5)
├── tests/              # Tests unitaires (preuve de rigueur)
├── LICENSE             # Ton fichier GPL v3
├── README.md           # Ta vitrine (Explications + Badges)
├── requirements.txt    # Liste des librairies (RDKit, MLflow, etc.)
└── .gitignore          # Pour ne pas push les données de 5 Go !


















#### License

This project follows a dual-licensing strategy:

    Software & Code: All source code (.py, .ipynb, scripts) is licensed under the GNU GPL v3.

    Documentation & Analysis: All descriptive text, methodologies, and visual results are licensed under CC BY-NC-SA 4.0. For commercial inquiries or custom deployment of this pipeline, please contact me via Malt or LinkedIn