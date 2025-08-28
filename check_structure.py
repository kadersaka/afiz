"""
Script de diagnostic pour votre structure de projet.
Structure : modules ML à la racine, modèles dans models/
"""

import os
import joblib
import json
from pathlib import Path


def check_project_structure():
    """Vérifie la structure du projet."""
    print("=== VÉRIFICATION DE LA STRUCTURE ===")
    print(f"Répertoire de travail: {os.getcwd()}")

    # Fichiers Python attendus à la racine
    python_files = [
        'app.py',
        'ml_predictor.py',
        'data_transformer.py',
        'train_models.py',
        'test_models.py'
    ]

    print("\n--- Fichiers Python ---")
    for file in python_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'✅ Existe' if exists else '❌ Manquant'}")

    # Répertoires
    directories = ['models', 'data']
    print("\n--- Répertoires ---")
    for dir_name in directories:
        exists = os.path.exists(dir_name)
        print(f"  {dir_name}/: {'✅ Existe' if exists else '❌ Manquant'}")

        if exists:
            files = os.listdir(dir_name)
            print(f"    Contenu: {files}")

    return os.path.exists('models') and os.path.exists('ml_predictor.py')


def check_models_files():
    """Vérifie les fichiers de modèles."""
    print("\n=== VÉRIFICATION DES MODÈLES ===")
    models_dir = "models"

    if not os.path.exists(models_dir):
        print(f"❌ Répertoire {models_dir}/ manquant")
        return False

    required_files = [
        "knn_model.pkl",
        "rf_model.pkl",
        "label_mappings.pkl",
        "model_metadata.json"
    ]

    all_present = True
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0

        print(f"  {file}: {'✅' if exists else '❌'} ({size} bytes)")

        if not exists:
            all_present = False

    return all_present


def check_label_mappings():
    """Vérifie le contenu des mappings."""
    print("\n=== VÉRIFICATION DES MAPPINGS ===")

    mappings_file = "models/label_mappings.pkl"

    if not os.path.exists(mappings_file):
        print("❌ Fichier label_mappings.pkl manquant")
        return False

    try:
        mappings = joblib.load(mappings_file)
        print(f"✅ Mappings chargés")
        print(f"Variables trouvées: {list(mappings.keys())}")

        # Variables attendues (sans le préfixe Filiere pour les features)
        expected_features = ['aff_dom', 'moy_sci', 'moy_ann', 'sect_parent', 'part_act', 'acc_empl']

        print("\n--- Vérification des variables features ---")
        missing_features = []

        for var in expected_features:
            if var in mappings:
                print(f"✅ {var}: {len(mappings[var])} valeurs")
                # Afficher quelques exemples
                for key, value in list(mappings[var].items())[:3]:
                    print(f"    {key} -> {value}")
            else:
                print(f"❌ {var}: MANQUANT")
                missing_features.append(var)

        # Vérification de la variable cible
        if 'Filiere' in mappings:
            print(f"✅ Filiere (target): {len(mappings['Filiere'])} classes")
            for key, value in mappings['Filiere'].items():
                print(f"    {key} -> {value}")
        else:
            print("❌ Filiere (target): MANQUANT")
            missing_features.append('Filiere')

        return len(missing_features) == 0

    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return False


def test_ml_modules():
    """Teste l'importation des modules ML."""
    print("\n=== TEST DES MODULES ML ===")

    # Test ml_predictor
    try:
        from ml_predictor import RecommendationEngine
        print("✅ ml_predictor importé")

        # Test d'initialisation
        try:
            engine = RecommendationEngine(models_dir="models")
            print("✅ RecommendationEngine initialisé")

            # Test des informations du modèle
            try:
                info = engine.get_model_info()
                print(
                    f"✅ Modèles chargés - KNN: {info.get('knn_accuracy', 'N/A')}, RF: {info.get('rf_accuracy', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Info modèles: {e}")

        except Exception as e:
            print(f"❌ Erreur initialisation RecommendationEngine: {e}")

    except ImportError as e:
        print(f"❌ Erreur import ml_predictor: {e}")

    # Test data_transformer
    try:
        from data_transformer import DataTransformer
        print("✅ data_transformer importé")

        transformer = DataTransformer()
        questions = transformer.get_questions()
        print(f"✅ DataTransformer initialisé - {len(questions)} questions")

    except ImportError as e:
        print(f"❌ Erreur import data_transformer: {e}")


def test_complete_workflow():
    """Teste le workflow complet."""
    print("\n=== TEST DU WORKFLOW COMPLET ===")

    try:
        from ml_predictor import RecommendationEngine
        from data_transformer import DataTransformer

        engine = RecommendationEngine(models_dir="models")
        transformer = DataTransformer()

        # Données de test
        test_data = {
            'aff_dom': 'Domaine Industriel',
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        print("Test de validation...")
        validation = transformer.validate_user_input(test_data)

        if validation['is_valid']:
            print("✅ Validation réussie")

            print("Test de prédiction...")
            recommendation = engine.get_recommendation(validation['cleaned_data'])
            print(f"✅ Prédiction réussie: {recommendation['final_recommendation']}")
            print(f"   Confiance: {recommendation['confidence']}")

        else:
            print(f"❌ Validation échouée: {validation['errors']}")

    except Exception as e:
        print(f"❌ Erreur workflow: {e}")
        import traceback
        traceback.print_exc()


def provide_solutions():
    """Fournit des solutions selon les problèmes détectés."""
    print("\n=== SOLUTIONS ===")

    if not os.path.exists('models'):
        print("🔧 Répertoire models/ manquant:")
        print("   mkdir models")

    if not os.path.exists('models/knn_model.pkl'):
        print("🔧 Modèles manquants:")
        print("   python train_models.py")

    if not os.path.exists('ml_predictor.py'):
        print("🔧 Module ml_predictor.py manquant:")
        print("   Copiez le fichier ml_predictor.py dans la racine du projet")

    if not os.path.exists('data'):
        print("🔧 Répertoire data/ manquant:")
        print("   mkdir data")
        print("   # Placez quest_data.xlsx dans data/")


def main():
    """Exécute le diagnostic complet."""
    print("🔍 DIAGNOSTIC DE LA STRUCTURE DU PROJET\n")

    structure_ok = check_project_structure()
    models_ok = check_models_files()
    mappings_ok = check_label_mappings()

    if structure_ok and models_ok and mappings_ok:
        test_ml_modules()
        test_complete_workflow()
    else:
        provide_solutions()

    print(f"\n{'=' * 50}")
    print("RÉSUMÉ DU DIAGNOSTIC")
    print(f"{'=' * 50}")
    print(f"Structure du projet: {'✅' if structure_ok else '❌'}")
    print(f"Fichiers de modèles: {'✅' if models_ok else '❌'}")
    print(f"Mappings valides: {'✅' if mappings_ok else '❌'}")

    if structure_ok and models_ok and mappings_ok:
        print("\n🎉 Tout semble en ordre ! Vous pouvez lancer :")
        print("   python app.py --port 5001")
    else:
        print("\n⚠️  Des corrections sont nécessaires avant de lancer l'application.")


if __name__ == "__main__":
    main()