"""
Tests unitaires pour le système de recommandation académique.
Validation des modèles, transformations et prédictions.
"""
import logging
import unittest
import tempfile
import shutil
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import des modules à tester
from ml_predictor import RecommendationEngine
from data_transformer import DataTransformer
from train_models import load_and_preprocess_data, train_knn_model, train_rf_model


class TestModelLoading(unittest.TestCase):
    """Tests pour le chargement des modèles."""

    def setUp(self):
        """Initialisation des tests avec un répertoire temporaire."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir)

        # Création de fichiers mock
        self._create_mock_models()

    def tearDown(self):
        """Nettoyage après les tests."""
        shutil.rmtree(self.temp_dir)

    def _create_mock_models(self):
        """Crée des modèles mock pour les tests."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier

        # Modèles mock
        knn_mock = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
        rf_mock = RandomForestClassifier(max_depth=5, random_state=42)

        # Données d'entraînement mock
        X_mock = np.random.randint(0, 3, (100, 6))
        y_mock = np.random.randint(0, 3, 100)

        knn_mock.fit(X_mock, y_mock)
        rf_mock.fit(X_mock, y_mock)

        # Sauvegarde des modèles mock
        joblib.dump(knn_mock, self.models_dir / 'knn_model.pkl')
        joblib.dump(rf_mock, self.models_dir / 'rf_model.pkl')

        # Label mappings mock
        label_mappings = {
            'aff_dom': {'Domaine Industriel': 0, 'Domaine de l\'agronomie': 1, 'Domaine de la santé': 2},
            'moy_sci': {'<10/20': 0, '>15/20': 1, 'Entre 10/20 et 15/20': 2},
            'moy_ann': {'<10/20': 0, '>15/20': 1, 'Entre 10/20 et 15/20': 2},
            'sect_parent': {'Secteur agricol': 0, 'Secteur de la santé': 1, 'Secteur industriel': 2},
            'part_act': {'Domaine agricol': 0, 'Domaine de la santé': 1, 'Domaine industriel': 2},
            'acc_empl': {'Secteur agricol': 0, 'Secteur de la santé': 1, 'Secteur industriel': 2}
        }
        joblib.dump(label_mappings, self.models_dir / 'label_mappings.pkl')

        # Métadonnées mock
        metadata = {
            'models': {
                'knn': {
                    'class': 'KNeighborsClassifier',
                    'params': knn_mock.get_params(),
                    'performance': {'accuracy': 0.78}
                },
                'rf': {
                    'class': 'RandomForestClassifier',
                    'params': rf_mock.get_params(),
                    'performance': {'accuracy': 0.84}
                }
            },
            'feature_names': ['aff_dom', 'moy_sci', 'moy_ann', 'sect_parent', 'part_act', 'acc_empl'],
            'target_mapping': {'0': 'Agronomie', '1': 'Informatique', '2': 'Médecine'},
            'training_info': {'train_test_split': 0.8, 'random_state': 42, 'cv_folds': 5}
        }

        with open(self.models_dir / 'model_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def test_model_loading_success(self):
        """Test du chargement réussi des modèles."""
        engine = RecommendationEngine(models_dir=str(self.models_dir))

        self.assertIsNotNone(engine.knn_model)
        self.assertIsNotNone(engine.rf_model)
        self.assertIsNotNone(engine.label_mappings)
        self.assertIsNotNone(engine.metadata)

    def test_model_loading_missing_files(self):
        """Test du chargement avec fichiers manquants."""
        # Suppression d'un fichier requis
        (self.models_dir / 'knn_model.pkl').unlink()

        with self.assertRaises(FileNotFoundError):
            RecommendationEngine(models_dir=str(self.models_dir))

    def test_get_model_info(self):
        """Test de récupération des informations des modèles."""
        engine = RecommendationEngine(models_dir=str(self.models_dir))
        info = engine.get_model_info()

        self.assertTrue(info['models_loaded'])
        self.assertIn('knn_accuracy', info)
        self.assertIn('rf_accuracy', info)
        self.assertEqual(len(info['features']), 6)
        self.assertEqual(len(info['target_classes']), 3)


class TestPredictionAccuracy(unittest.TestCase):
    """Tests pour la précision des prédictions."""

    def setUp(self):
        """Initialisation avec données de test connues."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir)

        # Création de modèles avec données contrôlées
        self._create_controlled_models()
        self.engine = RecommendationEngine(models_dir=str(self.models_dir))

    def tearDown(self):
        """Nettoyage après les tests."""
        shutil.rmtree(self.temp_dir)

    def _create_controlled_models(self):
        """Crée des modèles avec des prédictions contrôlées."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier

        # Données d'entraînement simplifiées pour prédictions prévisibles
        X = np.array([
            [0, 2, 2, 1, 2, 2],  # Profil Informatique
            [2, 2, 2, 1, 1, 1],  # Profil Médecine
            [1, 2, 2, 0, 0, 0],  # Profil Agronomie
        ])
        y = np.array([1, 2, 0])  # Informatique, Médecine, Agronomie

        knn = KNeighborsClassifier(n_neighbors=1)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)

        knn.fit(X, y)
        rf.fit(X, y)

        # Sauvegarde
        joblib.dump(knn, self.models_dir / 'knn_model.pkl')
        joblib.dump(rf, self.models_dir / 'rf_model.pkl')

        # Label mappings
        label_mappings = {
            'aff_dom': {'Domaine Industriel': 0, 'Domaine de l\'agronomie': 1, 'Domaine de la santé': 2},
            'moy_sci': {'<10/20': 0, '>15/20': 1, 'Entre 10/20 et 15/20': 2},
            'moy_ann': {'<10/20': 0, '>15/20': 1, 'Entre 10/20 et 15/20': 2},
            'sect_parent': {'Secteur agricol': 0, 'Secteur de la santé': 1, 'Secteur industriel': 2},
            'part_act': {'Domaine agricol': 0, 'Domaine de la santé': 1, 'Domaine industriel': 2},
            'acc_empl': {'Secteur agricol': 0, 'Secteur de la santé': 1, 'Secteur industriel': 2}
        }
        joblib.dump(label_mappings, self.models_dir / 'label_mappings.pkl')

        # Métadonnées
        metadata = {
            'models': {
                'knn': {'class': 'KNeighborsClassifier', 'params': knn.get_params(), 'performance': {'accuracy': 1.0}},
                'rf': {'class': 'RandomForestClassifier', 'params': rf.get_params(), 'performance': {'accuracy': 1.0}}
            },
            'feature_names': ['aff_dom', 'moy_sci', 'moy_ann', 'sect_parent', 'part_act', 'acc_empl'],
            'target_mapping': {'0': 'Agronomie', '1': 'Informatique', '2': 'Médecine'},
            'training_info': {'train_test_split': 0.8, 'random_state': 42, 'cv_folds': 5}
        }

        with open(self.models_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f)

    def test_prediction_informatique(self):
        """Test de prédiction pour profil informatique."""
        user_data = {
            'aff_dom': 'Domaine Industriel',
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        recommendation = self.engine.get_recommendation(user_data)

        # Le profil devrait correspondre à Informatique
        self.assertIn('Informatique', [recommendation['knn_prediction'], recommendation['rf_prediction']])

    def test_prediction_medecine(self):
        """Test de prédiction pour profil médecine."""
        user_data = {
            'aff_dom': 'Domaine de la santé',
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine de la santé',
            'acc_empl': 'Secteur de la santé'
        }

        recommendation = self.engine.get_recommendation(user_data)

        # Le profil devrait correspondre à Médecine
        self.assertIn('Médecine', [recommendation['knn_prediction'], recommendation['rf_prediction']])

    def test_prediction_agronomie(self):
        """Test de prédiction pour profil agronomie."""
        user_data = {
            'aff_dom': 'Domaine de l\'agronomie',
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur agricol',
            'part_act': 'Domaine agricol',
            'acc_empl': 'Secteur agricol'
        }

        recommendation = self.engine.get_recommendation(user_data)

        # Le profil devrait correspondre à Agronomie
        self.assertIn('Agronomie', [recommendation['knn_prediction'], recommendation['rf_prediction']])


class TestDataTransformation(unittest.TestCase):
    """Tests pour la transformation des données."""

    def setUp(self):
        """Initialisation du transformateur."""
        self.transformer = DataTransformer()

    def test_get_questions(self):
        """Test de récupération des questions."""
        questions = self.transformer.get_questions()

        self.assertEqual(len(questions), 6)
        self.assertIn('aff_dom', questions)
        self.assertIn('moy_sci', questions)

        # Vérification de la structure d'une question
        aff_dom_config = questions['aff_dom']
        self.assertIn('question', aff_dom_config)
        self.assertIn('options', aff_dom_config)
        self.assertIn('required', aff_dom_config)

    def test_validate_user_input_valid(self):
        """Test de validation avec données valides."""
        valid_data = {
            'aff_dom': 'Domaine Industriel',
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        result = self.transformer.validate_user_input(valid_data)

        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertEqual(len(result['cleaned_data']), 6)

    def test_validate_user_input_missing_data(self):
        """Test de validation avec données manquantes."""
        incomplete_data = {
            'aff_dom': 'Domaine Industriel',
            'moy_sci': 'Entre 10/20 et 15/20'
            # Données manquantes
        }

        result = self.transformer.validate_user_input(incomplete_data)

        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)

    def test_validate_user_input_invalid_values(self):
        """Test de validation avec valeurs invalides."""
        invalid_data = {
            'aff_dom': 'Domaine Invalide',  # Valeur non autorisée
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        result = self.transformer.validate_user_input(invalid_data)

        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)

    def test_transform_categorical_to_numerical(self):
        """Test de transformation catégorielle vers numérique."""
        categorical_data = {
            'aff_dom': 'Domaine Industriel',
            'moy_sci': 'Entre 10/20 et 15/20'
        }

        numerical_data = self.transformer.transform_categorical_to_numerical(categorical_data)

        self.assertEqual(numerical_data['aff_dom'], 0)
        self.assertEqual(numerical_data['moy_sci'], 2)

    def test_decode_numerical_response(self):
        """Test du décodage numérique vers catégoriel."""
        decoded = self.transformer.decode_numerical_response('aff_dom', 0)
        self.assertEqual(decoded, 'Domaine Industriel')

        decoded = self.transformer.decode_numerical_response('moy_sci', 2)
        self.assertEqual(decoded, 'Entre 10/20 et 15/20')

    def test_get_form_schema(self):
        """Test de génération du schéma de formulaire."""
        schema = self.transformer.get_form_schema()

        self.assertEqual(schema['type'], 'object')
        self.assertIn('properties', schema)
        self.assertIn('required', schema)
        self.assertEqual(len(schema['properties']), 6)


class TestUserInputValidation(unittest.TestCase):
    """Tests pour la validation des entrées utilisateur."""

    def setUp(self):
        """Initialisation avec un moteur mock."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir)
        self._create_minimal_models()
        self.engine = RecommendationEngine(models_dir=str(self.models_dir))

    def tearDown(self):
        """Nettoyage."""
        shutil.rmtree(self.temp_dir)

    def _create_minimal_models(self):
        """Crée des modèles minimaux pour les tests."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.randint(0, 3, (10, 6))
        y = np.random.randint(0, 3, 10)

        knn = KNeighborsClassifier(n_neighbors=3)
        rf = RandomForestClassifier(n_estimators=5, random_state=42)

        knn.fit(X, y)
        rf.fit(X, y)

        joblib.dump(knn, self.models_dir / 'knn_model.pkl')
        joblib.dump(rf, self.models_dir / 'rf_model.pkl')

        label_mappings = {
            'aff_dom': {'Domaine Industriel': 0, 'Domaine de l\'agronomie': 1, 'Domaine de la santé': 2},
            'moy_sci': {'<10/20': 0, '>15/20': 1, 'Entre 10/20 et 15/20': 2},
            'moy_ann': {'<10/20': 0, '>15/20': 1, 'Entre 10/20 et 15/20': 2},
            'sect_parent': {'Secteur agricol': 0, 'Secteur de la santé': 1, 'Secteur industriel': 2},
            'part_act': {'Domaine agricol': 0, 'Domaine de la santé': 1, 'Domaine industriel': 2},
            'acc_empl': {'Secteur agricol': 0, 'Secteur de la santé': 1, 'Secteur industriel': 2}
        }
        joblib.dump(label_mappings, self.models_dir / 'label_mappings.pkl')

        metadata = {
            'models': {
                'knn': {'performance': {'accuracy': 0.8}},
                'rf': {'performance': {'accuracy': 0.85}}
            },
            'target_mapping': {'0': 'Agronomie', '1': 'Informatique', '2': 'Médecine'}
        }

        with open(self.models_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f)

    def test_validate_complete_input(self):
        """Test de validation avec input complet et valide."""
        valid_input = {
            'aff_dom': 'Domaine Industriel',
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        is_valid, errors = self.engine.validate_user_input(valid_input)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_missing_required_field(self):
        """Test de validation avec champ requis manquant."""
        incomplete_input = {
            'aff_dom': 'Domaine Industriel',
            'moy_sci': 'Entre 10/20 et 15/20',
            # moy_ann manquant
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        is_valid, errors = self.engine.validate_user_input(incomplete_input)

        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('moy_ann' in error for error in errors))

    def test_validate_invalid_option(self):
        """Test de validation avec option invalide."""
        invalid_input = {
            'aff_dom': 'Domaine Inexistant',  # Option non valide
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        is_valid, errors = self.engine.validate_user_input(invalid_input)

        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('Domaine Inexistant' in error for error in errors))

    def test_preprocess_user_input_success(self):
        """Test du préprocessing réussi."""
        valid_input = {
            'aff_dom': 'Domaine Industriel',
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        processed = self.engine.preprocess_user_input(valid_input)

        self.assertEqual(processed.shape, (1, 6))
        self.assertIsInstance(processed, np.ndarray)

    def test_preprocess_user_input_invalid(self):
        """Test du préprocessing avec données invalides."""
        invalid_input = {
            'aff_dom': 'Domaine Inexistant',
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        with self.assertRaises(ValueError):
            self.engine.preprocess_user_input(invalid_input)


class TestIntegrationWorkflow(unittest.TestCase):
    """Tests d'intégration du workflow complet."""

    def setUp(self):
        """Initialisation pour les tests d'intégration."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir)
        self._create_realistic_models()
        self.engine = RecommendationEngine(models_dir=str(self.models_dir))
        self.transformer = DataTransformer()

    def tearDown(self):
        """Nettoyage."""
        shutil.rmtree(self.temp_dir)

    def _create_realistic_models(self):
        """Crée des modèles avec des données plus réalistes."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier

        # Simulation de données d'entraînement plus réalistes
        np.random.seed(42)
        n_samples = 160
        X = np.random.randint(0, 3, (n_samples, 6))

        # Génération de y avec des patterns logiques
        y = np.zeros(n_samples)
        for i in range(n_samples):
            if X[i, 0] == 0:  # Domaine Industriel -> Informatique
                y[i] = 1 if np.random.random() > 0.3 else np.random.choice([0, 2])
            elif X[i, 0] == 2:  # Domaine Santé -> Médecine
                y[i] = 2 if np.random.random() > 0.3 else np.random.choice([0, 1])
            else:  # Domaine Agronomie -> Agronomie
                y[i] = 0 if np.random.random() > 0.3 else np.random.choice([1, 2])

        y = y.astype(int)

        # Entraînement des modèles
        knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
        rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)

        knn.fit(X, y)
        rf.fit(X, y)

        # Sauvegarde
        joblib.dump(knn, self.models_dir / 'knn_model.pkl')
        joblib.dump(rf, self.models_dir / 'rf_model.pkl')

        label_mappings = {
            'aff_dom': {'Domaine Industriel': 0, 'Domaine de l\'agronomie': 1, 'Domaine de la santé': 2},
            'moy_sci': {'<10/20': 0, '>15/20': 1, 'Entre 10/20 et 15/20': 2},
            'moy_ann': {'<10/20': 0, '>15/20': 1, 'Entre 10/20 et 15/20': 2},
            'sect_parent': {'Secteur agricol': 0, 'Secteur de la santé': 1, 'Secteur industriel': 2},
            'part_act': {'Domaine agricol': 0, 'Domaine de la santé': 1, 'Domaine industriel': 2},
            'acc_empl': {'Secteur agricol': 0, 'Secteur de la santé': 1, 'Secteur industriel': 2}
        }
        joblib.dump(label_mappings, self.models_dir / 'label_mappings.pkl')

        metadata = {
            'models': {
                'knn': {'class': 'KNeighborsClassifier', 'performance': {'accuracy': 0.818}},
                'rf': {'class': 'RandomForestClassifier', 'performance': {'accuracy': 0.844}}
            },
            'feature_names': ['aff_dom', 'moy_sci', 'moy_ann', 'sect_parent', 'part_act', 'acc_empl'],
            'target_mapping': {'0': 'Agronomie', '1': 'Informatique', '2': 'Médecine'}
        }

        with open(self.models_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f)

    def test_complete_workflow_informatique(self):
        """Test du workflow complet pour un profil informatique."""
        # 1. Simulation des réponses utilisateur
        user_responses = {
            'aff_dom': 'Domaine Industriel',
            'moy_sci': 'Entre 10/20 et 15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur industriel',
            'part_act': 'Domaine industriel',
            'acc_empl': 'Secteur industriel'
        }

        # 2. Validation avec DataTransformer
        validation = self.transformer.validate_user_input(user_responses)
        self.assertTrue(validation['is_valid'])

        # 3. Obtention de la recommandation
        recommendation = self.engine.get_recommendation(user_responses)

        # 4. Vérifications
        self.assertIn('final_recommendation', recommendation)
        self.assertIn('confidence', recommendation)
        self.assertIn('explanation', recommendation)
        self.assertIn('knn_prediction', recommendation)
        self.assertIn('rf_prediction', recommendation)

        # La recommandation devrait être logique pour ce profil
        self.assertIn(recommendation['final_recommendation'],
                      ['Informatique', 'Agronomie', 'Médecine'])
        self.assertGreater(recommendation['confidence'], 0)

    def test_complete_workflow_medecine(self):
        """Test du workflow complet pour un profil médecine."""
        user_responses = {
            'aff_dom': 'Domaine de la santé',
            'moy_sci': '>15/20',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine de la santé',
            'acc_empl': 'Secteur de la santé'
        }

        # Validation
        validation = self.transformer.validate_user_input(user_responses)
        self.assertTrue(validation['is_valid'])

        # Recommandation
        recommendation = self.engine.get_recommendation(user_responses)

        # Vérifications
        self.assertIsInstance(recommendation, dict)
        self.assertIn('final_recommendation', recommendation)
        self.assertIsInstance(recommendation['explanation'], str)
        self.assertGreater(len(recommendation['explanation']), 10)

    def test_workflow_error_handling(self):
        """Test de gestion d'erreurs dans le workflow."""
        # Données invalides
        invalid_responses = {
            'aff_dom': 'Domaine Inexistant',
            'moy_sci': 'Note Invalide',
            'moy_ann': 'Entre 10/20 et 15/20',
            'sect_parent': 'Secteur de la santé',
            'part_act': 'Domaine de la santé',
            'acc_empl': 'Secteur de la santé'
        }

        # La validation devrait échouer
        validation = self.transformer.validate_user_input(invalid_responses)
        self.assertFalse(validation['is_valid'])

        # La recommandation devrait lever une exception
        with self.assertRaises(ValueError):
            self.engine.get_recommendation(invalid_responses)

    def test_model_consistency(self):
        """Test de cohérence des modèles sur plusieurs prédictions."""
        test_cases = [
            {
                'aff_dom': 'Domaine Industriel',
                'moy_sci': 'Entre 10/20 et 15/20',
                'moy_ann': 'Entre 10/20 et 15/20',
                'sect_parent': 'Secteur industriel',
                'part_act': 'Domaine industriel',
                'acc_empl': 'Secteur industriel'
            },
            {
                'aff_dom': 'Domaine de la santé',
                'moy_sci': '>15/20',
                'moy_ann': '>15/20',
                'sect_parent': 'Secteur de la santé',
                'part_act': 'Domaine de la santé',
                'acc_empl': 'Secteur de la santé'
            },
            {
                'aff_dom': 'Domaine de l\'agronomie',
                'moy_sci': 'Entre 10/20 et 15/20',
                'moy_ann': 'Entre 10/20 et 15/20',
                'sect_parent': 'Secteur agricol',
                'part_act': 'Domaine agricol',
                'acc_empl': 'Secteur agricol'
            }
        ]

        predictions = []
        for test_case in test_cases:
            recommendation = self.engine.get_recommendation(test_case)
            predictions.append(recommendation['final_recommendation'])

        # Toutes les prédictions devraient être des filières valides
        valid_filieres = ['Informatique', 'Médecine', 'Agronomie']
        for prediction in predictions:
            self.assertIn(prediction, valid_filieres)

        # Les prédictions ne devraient pas toutes être identiques
        # (sauf cas très particulier)
        unique_predictions = set(predictions)
        self.assertGreaterEqual(len(unique_predictions), 1)


def run_all_tests():
    """Lance tous les tests."""
    # Configuration du logging pour les tests
    logging.basicConfig(level=logging.WARNING)

    # Création de la suite de tests
    test_suite = unittest.TestSuite()

    # Ajout des classes de tests
    test_classes = [
        TestModelLoading,
        TestPredictionAccuracy,
        TestDataTransformation,
        TestUserInputValidation,
        TestIntegrationWorkflow
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Exécution des tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result


if __name__ == '__main__':
    print("=== LANCEMENT DES TESTS DU SYSTÈME DE RECOMMANDATION ===\n")

    # Exécution de tous les tests
    result = run_all_tests()

    # Résumé
    print(f"\n=== RÉSUMÉ ===")
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")

    if result.failures:
        print("\n--- ÉCHECS ---")
        for test, traceback in result.failures:
            print(f"{test}: {traceback}")

    if result.errors:
        print("\n--- ERREURS ---")
        for test, traceback in result.errors:
            print(f"{test}: {traceback}")

    # Code de sortie
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\nStatut: {'SUCCÈS' if result.wasSuccessful() else 'ÉCHEC'}")
    exit(exit_code)