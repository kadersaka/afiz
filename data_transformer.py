"""
Module de transformation des données pour le système de recommandation académique.
Utilitaires pour la validation et transformation des données utilisateur.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Classe utilitaire pour la transformation et validation des données utilisateur.
    """

    def __init__(self):
        """Initialise le transformateur avec les questions et options prédéfinies."""
        self.questions_config = self._load_questions_config()
        self.variable_mappings = self._load_variable_mappings()

    def _load_questions_config(self) -> Dict[str, Any]:
        """
        Définit la structure des questions pour le formulaire web.

        Returns:
            Configuration des questions avec options et métadonnées
        """
        return {
            'aff_dom': {
                'question': "Avec quel domaine avez-vous des affinités ?",
                'type': 'select',
                'required': True,
                'options': [
                    'Domaine Industriel',
                    'Domaine de l\'agronomie',
                    'Domaine de la santé'
                ],
                'help_text': "Choisissez le domaine qui vous attire le plus professionnellement."
            },
            'moy_sci': {
                'question': "Dans quel intervalle se situent généralement vos moyennes en sciences ?",
                'type': 'select',
                'required': True,
                'options': [
                    '<10/20',
                    'Entre 10/20 et 15/20',
                    '>15/20'
                ],
                'help_text': "Moyennes en mathématiques, sciences physiques, SVT, informatique."
            },
            'moy_ann': {
                'question': "Dans quel intervalle se situent vos moyennes annuelles (2nde à Terminale) ?",
                'type': 'select',
                'required': True,
                'options': [
                    '<10/20',
                    'Entre 10/20 et 15/20',
                    '>15/20'
                ],
                'help_text': "Moyennes générales sur l'ensemble de vos années lycée."
            },
            'sect_parent': {
                'question': "Dans quel secteur avez-vous des proches exerçant un métier ?",
                'type': 'select',
                'required': True,
                'options': [
                    'Secteur agricol',
                    'Secteur de la santé',
                    'Secteur industriel'
                ],
                'help_text': "Secteur professionnel de vos parents, amis ou proches."
            },
            'part_act': {
                'question': "Dans quel domaine avez-vous participé à des activités ?",
                'type': 'select',
                'required': True,
                'options': [
                    'Domaine agricol',
                    'Domaine de la santé',
                    'Domaine industriel'
                ],
                'help_text': "Activités auxquelles vous avez participé ou que vous avez menées."
            },
            'acc_empl': {
                'question': "Quel secteur offre la meilleure accessibilité au marché de l'emploi ?",
                'type': 'select',
                'required': True,
                'options': [
                    'Secteur agricol',
                    'Secteur de la santé',
                    'Secteur industriel'
                ],
                'help_text': "Selon vous, quel secteur offre le plus d'opportunités d'emploi."
            }
        }

    def _load_variable_mappings(self) -> Dict[str, Dict[str, int]]:
        """
        Charge les mappings entre valeurs textuelles et codes numériques.

        Returns:
            Dictionnaire des mappings pour chaque variable
        """
        return {
            'aff_dom': {
                'Domaine Industriel': 0,
                'Domaine de l\'agronomie': 1,
                'Domaine de la santé': 2
            },
            'moy_sci': {
                '<10/20': 0,
                '>15/20': 1,
                'Entre 10/20 et 15/20': 2
            },
            'moy_ann': {
                '<10/20': 0,
                '>15/20': 1,
                'Entre 10/20 et 15/20': 2
            },
            'sect_parent': {
                'Secteur agricol': 0,
                'Secteur de la santé': 1,
                'Secteur industriel': 2
            },
            'part_act': {
                'Domaine agricol': 0,
                'Domaine de la santé': 1,
                'Domaine industriel': 2
            },
            'acc_empl': {
                'Secteur agricol': 0,
                'Secteur de la santé': 1,
                'Secteur industriel': 2
            }
        }

    def get_questions(self) -> Dict[str, Any]:
        """
        Retourne la structure des questions pour le formulaire web.

        Returns:
            Dictionnaire avec toutes les questions et leurs options
        """
        return self.questions_config

    def get_question_options(self, question_id: str) -> List[str]:
        """
        Retourne les options disponibles pour une question spécifique.

        Args:
            question_id: Identifiant de la question

        Returns:
            Liste des options disponibles

        Raises:
            KeyError: Si l'ID de question n'existe pas
        """
        if question_id not in self.questions_config:
            raise KeyError(f"Question inconnue: {question_id}")

        return self.questions_config[question_id]['options']

    def validate_user_input(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Valide les données du formulaire utilisateur.

        Args:
            form_data: Données du formulaire web

        Returns:
            Dictionnaire avec le statut de validation et les erreurs éventuelles
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'cleaned_data': {}
        }

        try:
            # Vérification de la présence de toutes les questions requises
            required_questions = [q_id for q_id, config in self.questions_config.items()
                                  if config.get('required', False)]

            missing_questions = []
            for q_id in required_questions:
                if q_id not in form_data or not form_data[q_id].strip():
                    missing_questions.append(self.questions_config[q_id]['question'])

            if missing_questions:
                validation_result['is_valid'] = False
                validation_result['errors'].extend([
                    f"Question obligatoire non remplie: {q}" for q in missing_questions
                ])

            # Validation des valeurs pour chaque question
            for q_id, value in form_data.items():
                if q_id in self.questions_config:
                    # Nettoyage de la valeur
                    cleaned_value = value.strip()

                    # Vérification que la valeur est dans les options autorisées
                    valid_options = self.questions_config[q_id]['options']
                    if cleaned_value not in valid_options:
                        validation_result['is_valid'] = False
                        validation_result['errors'].append(
                            f"Valeur invalide '{cleaned_value}' pour la question '{q_id}'. "
                            f"Options valides: {valid_options}"
                        )
                    else:
                        validation_result['cleaned_data'][q_id] = cleaned_value
                else:
                    validation_result['warnings'].append(f"Question inconnue ignorée: {q_id}")

            logger.info(f"Validation terminée. Valide: {validation_result['is_valid']}")

        except Exception as e:
            logger.error(f"Erreur lors de la validation: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Erreur de validation: {str(e)}")

        return validation_result

    def transform_categorical_to_numerical(self, responses: Dict[str, str]) -> Dict[str, int]:
        """
        Transforme les réponses catégorielles en valeurs numériques.

        Args:
            responses: Réponses sous forme textuelle

        Returns:
            Réponses encodées numériquement

        Raises:
            ValueError: Si une valeur ne peut pas être encodée
        """
        try:
            numerical_responses = {}

            for variable, text_value in responses.items():
                if variable not in self.variable_mappings:
                    raise ValueError(f"Variable inconnue: {variable}")

                if text_value not in self.variable_mappings[variable]:
                    available_values = list(self.variable_mappings[variable].keys())
                    raise ValueError(f"Valeur '{text_value}' invalide pour {variable}. "
                                     f"Valeurs disponibles: {available_values}")

                numerical_responses[variable] = self.variable_mappings[variable][text_value]

            logger.info("Transformation catégorielle → numérique réussie")
            return numerical_responses

        except Exception as e:
            logger.error(f"Erreur lors de la transformation: {str(e)}")
            raise

    def get_form_schema(self) -> Dict[str, Any]:
        """
        Génère un schéma JSON pour le formulaire web (compatible avec des frameworks comme React).

        Returns:
            Schéma du formulaire au format JSON
        """
        schema = {
            'title': 'Questionnaire d\'orientation académique',
            'description': 'Répondez aux questions suivantes pour obtenir une recommandation personnalisée',
            'type': 'object',
            'properties': {},
            'required': []
        }

        for q_id, config in self.questions_config.items():
            schema['properties'][q_id] = {
                'title': config['question'],
                'type': 'string',
                'enum': config['options'],
                'description': config.get('help_text', '')
            }

            if config.get('required', False):
                schema['required'].append(q_id)

        return schema

    def get_reverse_mappings(self) -> Dict[str, Dict[int, str]]:
        """
        Retourne les mappings inversés (numérique → textuel).

        Returns:
            Dictionnaire des mappings inversés
        """
        reverse_mappings = {}
        for variable, mapping in self.variable_mappings.items():
            reverse_mappings[variable] = {v: k for k, v in mapping.items()}

        return reverse_mappings

    def decode_numerical_response(self, variable: str, numerical_value: int) -> str:
        """
        Décode une valeur numérique en texte original.

        Args:
            variable: Nom de la variable
            numerical_value: Valeur numérique à décoder

        Returns:
            Valeur textuelle correspondante

        Raises:
            ValueError: Si la valeur ne peut pas être décodée
        """
        if variable not in self.variable_mappings:
            raise ValueError(f"Variable inconnue: {variable}")

        reverse_mapping = {v: k for k, v in self.variable_mappings[variable].items()}

        if numerical_value not in reverse_mapping:
            raise ValueError(f"Valeur numérique {numerical_value} invalide pour {variable}")

        return reverse_mapping[numerical_value]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur la configuration des questions.

        Returns:
            Statistiques sur les questions et options
        """
        stats = {
            'total_questions': len(self.questions_config),
            'required_questions': sum(1 for config in self.questions_config.values()
                                      if config.get('required', False)),
            'total_options': sum(len(config['options']) for config in self.questions_config.values()),
            'questions_by_type': {}
        }

        # Statistiques par type de question
        type_counts = {}
        for config in self.questions_config.values():
            q_type = config.get('type', 'unknown')
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        stats['questions_by_type'] = type_counts

        return stats