"""
Module de prédiction pour le système de recommandation académique.
Classe RecommendationEngine pour effectuer des prédictions avec les modèles entraînés.
"""

import joblib
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Moteur de recommandation académique utilisant les modèles KNN et Random Forest.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialise le moteur de recommandation.

        Args:
            models_dir: Répertoire contenant les modèles sauvegardés
        """
        self.models_dir = Path(models_dir)
        self.knn_model = None
        self.rf_model = None
        self.label_mappings = None
        self.metadata = None
        self.target_mapping = {0: "Agronomie", 1: "Informatique", 2: "Médecine"}

        self._load_models()

    def _load_models(self) -> None:
        """Charge les modèles et métadonnées depuis les fichiers."""
        try:
            logger.info("Chargement des modèles...")

            # Vérification de l'existence des fichiers
            required_files = ['knn_model.pkl', 'rf_model.pkl', 'label_mappings.pkl', 'model_metadata.json']
            for file in required_files:
                if not (self.models_dir / file).exists():
                    raise FileNotFoundError(f"Fichier manquant: {file}")

            # Chargement des modèles
            self.knn_model = joblib.load(self.models_dir / 'knn_model.pkl')
            self.rf_model = joblib.load(self.models_dir / 'rf_model.pkl')
            self.label_mappings = joblib.load(self.models_dir / 'label_mappings.pkl')

            # Chargement des métadonnées
            with open(self.models_dir / 'model_metadata.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

            self.target_mapping = self.metadata['target_mapping']
            # Conversion des clés en int pour target_mapping
            self.target_mapping = {int(k): v for k, v in self.target_mapping.items()}

            logger.info("Modèles chargés avec succès")

        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {str(e)}")
            raise

    def preprocess_user_input(self, user_data: Dict[str, str]) -> np.ndarray:
        """
        Transforme les réponses utilisateur en format numérique pour les modèles ML.

        Args:
            user_data: Dictionnaire avec les réponses de l'utilisateur

        Returns:
            Array numpy avec les données encodées

        Raises:
            ValueError: Si les données utilisateur sont invalides
        """
        try:
            # Variables attendues (ordre important)
            expected_features = ['aff_dom', 'moy_sci', 'moy_ann', 'sect_parent', 'part_act', 'acc_empl']

            # Vérification de la présence de toutes les variables
            missing_features = [f for f in expected_features if f not in user_data]
            if missing_features:
                raise ValueError(f"Variables manquantes: {missing_features}")

            # Encodage des valeurs
            encoded_values = []
            for feature in expected_features:
                user_value = user_data[feature]

                # Vérification que la valeur existe dans les mappings
                if feature not in self.label_mappings:
                    raise ValueError(f"Mapping manquant pour la variable: {feature}")

                if user_value not in self.label_mappings[feature]:
                    available_values = list(self.label_mappings[feature].keys())
                    raise ValueError(f"Valeur invalide '{user_value}' pour {feature}. "
                                   f"Valeurs autorisées: {available_values}")

                encoded_value = self.label_mappings[feature][user_value]
                encoded_values.append(encoded_value)

            return np.array(encoded_values).reshape(1, -1)

        except Exception as e:
            logger.error(f"Erreur lors du préprocessing: {str(e)}")
            raise

    def predict_single(self, processed_data: np.ndarray) -> Dict[str, Any]:
        """
        Effectue une prédiction avec les deux modèles.

        Args:
            processed_data: Données préprocessées

        Returns:
            Dictionnaire avec les prédictions et probabilités
        """
        try:
            # Prédictions des deux modèles
            knn_pred = self.knn_model.predict(processed_data)[0]
            rf_pred = self.rf_model.predict(processed_data)[0]

            # Probabilités (si disponibles)
            knn_proba = None
            rf_proba = None

            try:
                knn_proba = self.knn_model.predict_proba(processed_data)[0]
                rf_proba = self.rf_model.predict_proba(processed_data)[0]
            except AttributeError:
                logger.warning("Probabilités non disponibles pour un des modèles")

            # Conversion en noms de filières
            knn_prediction = self.target_mapping[knn_pred]
            rf_prediction = self.target_mapping[rf_pred]

            results = {
                'knn_prediction': knn_prediction,
                'rf_prediction': rf_prediction,
                'knn_prediction_code': int(knn_pred),
                'rf_prediction_code': int(rf_pred)
            }

            # Ajout des probabilités si disponibles
            if knn_proba is not None and rf_proba is not None:
                # Moyenne des probabilités des deux modèles
                avg_proba = (knn_proba + rf_proba) / 2

                probabilities = {}
                for code, name in self.target_mapping.items():
                    probabilities[name] = float(avg_proba[code])

                results['probabilities'] = probabilities
                results['knn_probabilities'] = {self.target_mapping[i]: float(p) for i, p in enumerate(knn_proba)}
                results['rf_probabilities'] = {self.target_mapping[i]: float(p) for i, p in enumerate(rf_proba)}

            return results

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise

    def get_recommendation(self, user_responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Pipeline complète: transforme les réponses utilisateur en recommandation.

        Args:
            user_responses: Réponses du formulaire utilisateur

        Returns:
            Dictionnaire avec la recommandation complète et les explications
        """
        try:
            # 1. Préprocessing des données
            processed_data = self.preprocess_user_input(user_responses)

            # 2. Prédictions
            predictions = self.predict_single(processed_data)

            # 3. Détermination de la recommandation finale
            knn_pred = predictions['knn_prediction']
            rf_pred = predictions['rf_prediction']

            # Logique de consensus (vous pouvez l'adapter)
            if knn_pred == rf_pred:
                final_recommendation = knn_pred
                confidence = 0.9  # Haute confiance quand les deux modèles s'accordent
            else:
                # En cas de désaccord, utiliser le modèle avec la meilleure performance
                knn_accuracy = self.metadata['models']['knn']['performance']['accuracy']
                rf_accuracy = self.metadata['models']['rf']['performance']['accuracy']

                if knn_accuracy > rf_accuracy:
                    final_recommendation = knn_pred
                    confidence = 0.7
                else:
                    final_recommendation = rf_pred
                    confidence = 0.7

            # 4. Génération de l'explication
            explanation = self.explain_prediction(predictions, user_responses)

            # 5. Construction de la réponse finale
            recommendation = {
                'final_recommendation': final_recommendation,
                'confidence': confidence,
                'knn_prediction': knn_pred,
                'rf_prediction': rf_pred,
                'explanation': explanation,
                'model_agreement': knn_pred == rf_pred
            }

            # Ajout des probabilités si disponibles
            if 'probabilities' in predictions:
                recommendation['probabilities'] = predictions['probabilities']
                recommendation['detailed_probabilities'] = {
                    'knn': predictions['knn_probabilities'],
                    'rf': predictions['rf_probabilities']
                }

            logger.info(f"Recommandation générée: {final_recommendation} (confiance: {confidence})")
            return recommendation

        except Exception as e:
            logger.error(f"Erreur lors de la génération de recommandation: {str(e)}")
            raise

    def explain_prediction(self, predictions: Dict[str, Any], user_data: Dict[str, str]) -> str:
        """
        Génère une explication lisible de la prédiction.

        Args:
            predictions: Résultats de prédiction
            user_data: Données utilisateur originales

        Returns:
            Explication textuelle de la recommandation
        """
        try:
            knn_pred = predictions['knn_prediction']
            rf_pred = predictions['rf_prediction']

            # Analyse des affinités
            affinites = user_data.get('aff_dom', '')
            moyennes_sci = user_data.get('moy_sci', '')

            explanation_parts = []

            # Concordance des modèles
            if knn_pred == rf_pred:
                explanation_parts.append(f"Les deux modèles d'intelligence artificielle s'accordent pour recommander la filière **{knn_pred}**.")
            else:
                explanation_parts.append(f"Les modèles suggèrent deux options: {knn_pred} et {rf_pred}.")

            # Analyse basée sur les affinités
            affinity_explanations = {
                'Domaine Industriel': "Vos affinités avec le domaine industriel vous orientent naturellement vers l'informatique.",
                'Domaine de la santé': "Votre intérêt pour le domaine de la santé est un indicateur fort pour la médecine.",
                'Domaine de l\'agronomie': "Vos affinités agricoles correspondent parfaitement au profil agronomie."
            }

            if affinites in affinity_explanations:
                explanation_parts.append(affinity_explanations[affinites])

            # Analyse des performances académiques
            if moyennes_sci == '>15/20':
                explanation_parts.append("Vos excellentes moyennes en sciences renforcent cette recommandation.")
            elif moyennes_sci == 'Entre 10/20 et 15/20':
                explanation_parts.append("Vos moyennes en sciences sont solides et compatibles avec cette orientation.")

            # Ajout d'informations sur les probabilités si disponibles
            if 'probabilities' in predictions:
                probs = predictions['probabilities']
                max_prob_field = max(probs.keys(), key=lambda k: probs[k])
                max_prob = probs[max_prob_field]
                explanation_parts.append(f"La probabilité calculée pour {max_prob_field} est de {max_prob:.1%}.")

            return " ".join(explanation_parts)

        except Exception as e:
            logger.warning(f"Erreur lors de la génération d'explication: {str(e)}")
            return "Recommandation basée sur l'analyse de vos réponses par nos modèles d'intelligence artificielle."

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur les modèles chargés.

        Returns:
            Dictionnaire avec les métadonnées des modèles
        """
        if self.metadata is None:
            return {"error": "Modèles non chargés"}

        return {
            'models_loaded': True,
            'knn_accuracy': self.metadata['models']['knn']['performance']['accuracy'],
            'rf_accuracy': self.metadata['models']['rf']['performance']['accuracy'],
            'features': self.metadata['feature_names'],
            'target_classes': list(self.target_mapping.values())
        }

    def validate_user_input(self, user_data: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Valide les données utilisateur avant prédiction.

        Args:
            user_data: Données à valider

        Returns:
            Tuple (is_valid, error_messages)
        """
        errors = []

        # Variables requises
        required_features = ['aff_dom', 'moy_sci', 'moy_ann', 'sect_parent', 'part_act', 'acc_empl']

        # Vérification de la présence
        for feature in required_features:
            if feature not in user_data or not user_data[feature]:
                errors.append(f"Variable manquante: {feature}")

        # Vérification des valeurs autorisées
        for feature, value in user_data.items():
            if feature in self.label_mappings:
                if value not in self.label_mappings[feature]:
                    available = list(self.label_mappings[feature].keys())
                    errors.append(f"Valeur invalide '{value}' pour {feature}. Valeurs autorisées: {available}")

        return len(errors) == 0, errors