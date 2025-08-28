"""
Script d'entraînement des modèles ML pour le système de recommandation académique.
Transforme les données catégorielles et entraîne les modèles KNN et Random Forest.
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str = "data/quest_data.xlsx") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """
    Charge et prétraite les données depuis le fichier Excel.

    Args:
        data_path: Chemin vers le fichier de données

    Returns:
        X: Variables prédictives encodées
        y: Variable cible encodée
        encoders: Dictionnaire des mappings de labels
    """
    logger.info(f"Chargement des données depuis {data_path}")

    # Chargement des données
    data = pd.read_excel(data_path)

    # Renommage des colonnes
    var_names = {
        '1- Quelle est votre filière de formation ': 'Filiere',
        '2- Avec quel Domaine avez-vous des affinités? ': 'aff_dom',
        '3- Dans quel intervalle se situent généralement vos moyennes en sciences (maths, sciences physiques, sciences de la vie et de la terre, informatique) ? ': 'moy_sci',
        '4- Dans quel intervalle se situent généralement vos moyennes annuelles (de la 2nde en Tle) ? ': 'moy_ann',
        '5- Dans quel secteur avez-vous des proches (parents, amis, etc.) exerçant un métier ? ': 'sect_parent',
        '6- Dans quel domaine d\'activité avez-vous une fois participé ou mené une activités? ': 'part_act',
        '7- A quel secteur d\'activité, l\'accessibilité au marché de l\'emploi est le plus offrant ? ': 'acc_empl'
    }

    data.rename(columns=var_names, inplace=True)
    logger.info(f"Données chargées: {data.shape[0]} observations, {data.shape[1]} variables")

    # Encodage des variables catégorielles
    encoded_data = pd.DataFrame()
    label_mappings = {}

    logger.info("Encodage des variables catégorielles...")
    for col in data.columns:
        label_encoder = LabelEncoder()
        transformed_values = label_encoder.fit_transform(data[col])

        # Stocker les correspondances
        label_mappings[col] = dict(zip(label_encoder.classes_,
                                     label_encoder.transform(label_encoder.classes_)))

        encoded_data[col] = transformed_values

    # Séparation X et y
    X = encoded_data.drop("Filiere", axis=1)
    y = encoded_data["Filiere"]

    logger.info("Préprocessing terminé")
    return X, y, label_mappings

def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    """
    Entraîne le modèle KNN avec optimisation des hyperparamètres.

    Args:
        X_train: Données d'entraînement
        y_train: Labels d'entraînement

    Returns:
        Modèle KNN optimisé
    """
    logger.info("Entraînement du modèle KNN...")

    # Paramètres à tester
    param_grid_knn = {
        "n_neighbors": [2, 5, 10, 50],
        "weights": ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # GridSearch avec validation croisée
    knn = KNeighborsClassifier()
    grid_search_knn = GridSearchCV(
        knn,
        param_grid_knn,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )

    grid_search_knn.fit(X_train, y_train)

    logger.info(f"Meilleurs paramètres KNN: {grid_search_knn.best_params_}")
    logger.info(f"Meilleur score CV KNN: {grid_search_knn.best_score_:.4f}")

    return grid_search_knn.best_estimator_

def train_rf_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Entraîne le modèle Random Forest avec optimisation des hyperparamètres.

    Args:
        X_train: Données d'entraînement
        y_train: Labels d'entraînement

    Returns:
        Modèle Random Forest optimisé
    """
    logger.info("Entraînement du modèle Random Forest...")

    # Paramètres à tester
    param_grid_rf = {
        "n_estimators": [10, 100, 1000],
        "max_depth": [5, 7, 9]
    }

    # GridSearch avec validation croisée
    rf = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(
        rf,
        param_grid_rf,
        scoring="accuracy",
        cv=5
    )

    grid_search_rf.fit(X_train, y_train)

    logger.info(f"Meilleurs paramètres RF: {grid_search_rf.best_params_}")
    logger.info(f"Meilleur score CV RF: {grid_search_rf.best_score_:.4f}")

    return grid_search_rf.best_estimator_

def evaluate_models(knn_model: KNeighborsClassifier,
                   rf_model: RandomForestClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> Dict[str, Any]:
    """
    Évalue les performances des modèles sur les données de test.

    Args:
        knn_model: Modèle KNN entraîné
        rf_model: Modèle Random Forest entraîné
        X_test: Données de test
        y_test: Labels de test

    Returns:
        Dictionnaire avec les métriques d'évaluation
    """
    logger.info("Évaluation des modèles...")

    # Prédictions
    knn_pred = knn_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Calcul des métriques
    knn_accuracy = accuracy_score(y_test, knn_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    knn_confusion = confusion_matrix(y_test, knn_pred)
    rf_confusion = confusion_matrix(y_test, rf_pred)

    evaluation_results = {
        'knn': {
            'accuracy': knn_accuracy,
            'confusion_matrix': knn_confusion.tolist(),
            'classification_report': classification_report(y_test, knn_pred, output_dict=True)
        },
        'rf': {
            'accuracy': rf_accuracy,
            'confusion_matrix': rf_confusion.tolist(),
            'classification_report': classification_report(y_test, rf_pred, output_dict=True)
        }
    }

    logger.info(f"Accuracy KNN: {knn_accuracy:.4f}")
    logger.info(f"Accuracy RF: {rf_accuracy:.4f}")

    return evaluation_results

def save_models(knn_model: KNeighborsClassifier,
               rf_model: RandomForestClassifier,
               label_mappings: Dict[str, Dict],
               evaluation_results: Dict[str, Any],
               models_dir: str = "models") -> None:
    """
    Sauvegarde les modèles et métadonnées.

    Args:
        knn_model: Modèle KNN entraîné
        rf_model: Modèle Random Forest entraîné
        label_mappings: Mappings des labels
        evaluation_results: Résultats d'évaluation
        models_dir: Répertoire de sauvegarde
    """
    logger.info("Sauvegarde des modèles...")

    # Création du répertoire
    Path(models_dir).mkdir(exist_ok=True)

    # Sauvegarde des modèles
    joblib.dump(knn_model, f"{models_dir}/knn_model.pkl")
    joblib.dump(rf_model, f"{models_dir}/rf_model.pkl")
    joblib.dump(label_mappings, f"{models_dir}/label_mappings.pkl")

    # Création des métadonnées
    metadata = {
        'models': {
            'knn': {
                'class': 'KNeighborsClassifier',
                'params': knn_model.get_params(),
                'performance': evaluation_results['knn']
            },
            'rf': {
                'class': 'RandomForestClassifier',
                'params': rf_model.get_params(),
                'performance': evaluation_results['rf']
            }
        },
        'feature_names': ['aff_dom', 'moy_sci', 'moy_ann', 'sect_parent', 'part_act', 'acc_empl'],
        'target_mapping': {
            0: "Agronomie",
            1: "Informatique",
            2: "Médecine"
        },
        'training_info': {
            'train_test_split': 0.8,
            'random_state': 42,
            'cv_folds': 5
        }
    }

    # Sauvegarde des métadonnées
    with open(f"{models_dir}/model_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Modèles sauvegardés dans {models_dir}/")

def main():
    """Pipeline complète d'entraînement des modèles."""
    logger.info("=== DÉBUT DE L'ENTRAÎNEMENT ===")

    try:
        # 1. Chargement et préprocessing
        X, y, label_mappings = load_and_preprocess_data()

        # 2. Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Division train/test: {len(X_train)} / {len(X_test)}")

        # 3. Entraînement des modèles
        knn_model = train_knn_model(X_train, y_train)
        rf_model = train_rf_model(X_train, y_train)

        # 4. Évaluation
        evaluation_results = evaluate_models(knn_model, rf_model, X_test, y_test)

        # 5. Sauvegarde
        save_models(knn_model, rf_model, label_mappings, evaluation_results)

        logger.info("=== ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ===")

    except Exception as e:
        logger.error(f"Erreur durant l'entraînement: {str(e)}")
        raise

if __name__ == "__main__":
    main()