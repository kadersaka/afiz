"""
Application Flask pour le système de recommandation académique.
Adaptée pour la structure : modules ML à la racine, modèles dans models/
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
from datetime import datetime
import time
import os
from pathlib import Path

# Import des modules ML depuis la racine du projet
try:
    from ml_predictor import RecommendationEngine
    from data_transformer import DataTransformer
except ImportError as e:
    print(f"Erreur d'importation des modules ML: {e}")
    print("Assurez-vous que ml_predictor.py et data_transformer.py sont dans la racine du projet")
    exit(1)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation de l'application Flask
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Support des caractères UTF-8
CORS(app)  # Permet les requêtes cross-origin

# Variables globales pour les statistiques
stats = {
    'total_predictions': 0,
    'predictions_today': 0,
    'total_errors': 0,
    'start_time': datetime.now(),
    'response_times': [],
    'prediction_counts': {
        'Informatique': 0,
        'Médecine': 0,
        'Agronomie': 0
    }
}

# Initialisation des composants ML
try:
    logger.info("Initialisation du moteur de recommandation...")

    # Utilisation du répertoire models/ à la racine
    models_dir = "models"

    if not os.path.exists(models_dir):
        logger.warning(f"Répertoire des modèles non trouvé: {models_dir}")
        logger.info("Création du répertoire...")
        os.makedirs(models_dir, exist_ok=True)
        logger.warning("Vous devez entraîner les modèles avec: python train_models.py")

    # Vérification des fichiers requis
    required_files = ['knn_model.pkl', 'rf_model.pkl', 'label_mappings.pkl', 'model_metadata.json']
    missing_files = []

    for file in required_files:
        if not os.path.exists(os.path.join(models_dir, file)):
            missing_files.append(file)

    if missing_files:
        logger.error(f"Fichiers manquants dans {models_dir}/: {missing_files}")
        logger.error("Exécutez 'python train_models.py' pour créer les modèles")
        raise FileNotFoundError(f"Modèles manquants: {missing_files}")

    # Initialisation du moteur de recommandation
    engine = RecommendationEngine(models_dir=models_dir)
    transformer = DataTransformer()

    logger.info("✅ Système de recommandation initialisé avec succès")
    logger.info(f"📁 Modèles chargés depuis: {os.path.abspath(models_dir)}")

except Exception as e:
    logger.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
    logger.error("Traceback:", exc_info=True)
    engine = None
    transformer = None

# Décorateurs utilitaires
def require_models(f):
    """Décorateur pour vérifier que les modèles sont chargés."""
    def decorated_function(*args, **kwargs):
        if engine is None or transformer is None:
            return jsonify({
                'success': False,
                'error': 'Modèles non disponibles',
                'message': 'Les modèles ML ne sont pas chargés. Exécutez "python train_models.py" pour les créer.',
                'timestamp': datetime.now().isoformat(),
                'models_directory': os.path.abspath('models')
            }), 503
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def log_request_stats(f):
    """Décorateur pour logger les statistiques de requête."""
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            # Enregistrement du temps de réponse
            response_time = (time.time() - start_time) * 1000
            stats['response_times'].append(response_time)
            # Garder seulement les 1000 dernières mesures
            if len(stats['response_times']) > 1000:
                stats['response_times'] = stats['response_times'][-1000:]
            return result
        except Exception as e:
            stats['total_errors'] += 1
            raise
    decorated_function.__name__ = f.__name__
    return decorated_function

# Routes de l'API

@app.route('/', methods=['GET'])
def index():
    """Page d'accueil de l'API."""
    return jsonify({
        'message': 'API de Recommandation Académique',
        'version': '1.0.0',
        'status': 'active',
        'structure': {
            'models_directory': os.path.abspath('models'),
            'data_directory': os.path.abspath('data'),
            'ml_modules': 'racine du projet'
        },
        'endpoints': {
            'recommend': '/api/recommend',
            'questions': '/api/questions',
            'validate': '/api/validate',
            'health': '/api/health',
            'models_info': '/api/models/info',
            'stats': '/api/stats'
        },
        'models_loaded': engine is not None and transformer is not None
    })

@app.route('/api/recommend', methods=['POST'])
@require_models
@log_request_stats
def get_recommendation():
    """
    Endpoint principal pour obtenir une recommandation académique.
    """
    start_time = time.time()

    try:
        # Récupération des données JSON
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Format de données invalide',
                'message': 'Content-Type doit être application/json',
                'timestamp': datetime.now().isoformat()
            }), 400

        user_data = request.get_json()
        if not user_data:
            return jsonify({
                'success': False,
                'error': 'Données manquantes',
                'message': 'Le body de la requête ne peut pas être vide',
                'timestamp': datetime.now().isoformat()
            }), 400

        logger.info(f"Requête de recommandation reçue: {list(user_data.keys())}")

        # Validation préalable des données
        validation_result = transformer.validate_user_input(user_data)

        if not validation_result['is_valid']:
            return jsonify({
                'success': False,
                'error': 'Données invalides',
                'details': validation_result['errors'],
                'warnings': validation_result.get('warnings', []),
                'timestamp': datetime.now().isoformat()
            }), 400

        # Génération de la recommandation
        recommendation = engine.get_recommendation(validation_result['cleaned_data'])

        # Mise à jour des statistiques
        stats['total_predictions'] += 1
        stats['predictions_today'] += 1
        final_rec = recommendation['final_recommendation']
        if final_rec in stats['prediction_counts']:
            stats['prediction_counts'][final_rec] += 1

        # Calcul du temps de traitement
        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Recommandation générée: {final_rec} (temps: {processing_time:.2f}ms)")

        return jsonify({
            'success': True,
            'recommendation': recommendation,
            'processing_time_ms': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        }), 200

    except ValueError as ve:
        logger.warning(f"Erreur de validation: {str(ve)}")
        return jsonify({
            'success': False,
            'error': 'Données invalides',
            'message': str(ve),
            'timestamp': datetime.now().isoformat()
        }), 400

    except Exception as e:
        logger.error(f"Erreur lors de la génération de recommandation: {str(e)}")
        logger.error(traceback.format_exc())

        return jsonify({
            'success': False,
            'error': 'Erreur interne du serveur',
            'message': str(e),  # Plus de détails pour le debug
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/questions', methods=['GET'])
@require_models
def get_questions():
    """Récupère la structure des questions pour le formulaire."""
    try:
        questions = transformer.get_questions()

        return jsonify({
            'success': True,
            'questions': questions,
            'total_questions': len(questions),
            'required_questions': sum(1 for q in questions.values() if q.get('required', False))
        }), 200

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des questions: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Erreur interne',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/validate', methods=['POST'])
@require_models
def validate_data():
    """Valide les données utilisateur sans faire de prédiction."""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Format invalide',
                'message': 'Content-Type doit être application/json'
            }), 400

        user_data = request.get_json()
        validation_result = transformer.validate_user_input(user_data)

        message = "Données valides" if validation_result['is_valid'] else "Données invalides"

        return jsonify({
            'success': True,
            'validation': validation_result,
            'message': message
        }), 200

    except Exception as e:
        logger.error(f"Erreur lors de la validation: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Erreur interne',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/models/info', methods=['GET'])
@require_models
def get_models_info():
    """Récupère les informations sur les modèles chargés."""
    try:
        models_info = engine.get_model_info()

        # Ajout d'informations sur la structure
        models_info['file_info'] = {
            'models_directory': os.path.abspath('models'),
            'files_found': os.listdir('models') if os.path.exists('models') else []
        }

        return jsonify({
            'success': True,
            'models_info': models_info
        }), 200

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos modèles: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Erreur interne',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check de l'API."""
    try:
        uptime = datetime.now() - stats['start_time']

        # Vérification des fichiers de modèles
        models_files_status = {}
        models_dir = 'models'

        if os.path.exists(models_dir):
            required_files = ['knn_model.pkl', 'rf_model.pkl', 'label_mappings.pkl', 'model_metadata.json']
            for file in required_files:
                file_path = os.path.join(models_dir, file)
                models_files_status[file] = {
                    'exists': os.path.exists(file_path),
                    'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                }

        health_status = {
            'status': 'healthy' if (engine is not None and transformer is not None) else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'models_status': {
                'knn_loaded': engine is not None and hasattr(engine, 'knn_model') and engine.knn_model is not None,
                'rf_loaded': engine is not None and hasattr(engine, 'rf_model') and engine.rf_model is not None,
                'mappings_loaded': engine is not None and hasattr(engine, 'label_mappings') and engine.label_mappings is not None
            },
            'file_system': {
                'models_directory': os.path.abspath('models'),
                'models_files': models_files_status,
                'working_directory': os.getcwd()
            },
            'system_info': {
                'uptime_seconds': int(uptime.total_seconds()),
                'total_predictions': stats['total_predictions'],
                'total_errors': stats['total_errors']
            }
        }

        status_code = 200 if health_status['status'] == 'healthy' else 503

        return jsonify(health_status), status_code

    except Exception as e:
        logger.error(f"Erreur lors du health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

# Gestion d'erreurs globale
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint non trouvé',
        'message': 'L\'URL demandée n\'existe pas',
        'available_endpoints': [
            '/api/recommend',
            '/api/questions',
            '/api/validate',
            '/api/models/info',
            '/api/health'
        ],
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur interne du serveur: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Erreur interne du serveur',
        'message': 'Une erreur inattendue s\'est produite',
        'timestamp': datetime.now().isoformat()
    }), 500

# Point d'entrée de l'application
if __name__ == '__main__':
    # Vérification de l'environnement
    logger.info("=== DÉMARRAGE DE L'APPLICATION FLASK ===")
    logger.info(f"Répertoire de travail: {os.getcwd()}")
    logger.info(f"Répertoire des modèles: {os.path.abspath('models')}")
    logger.info(f"Moteur ML disponible: {engine is not None}")
    logger.info(f"Transformer disponible: {transformer is not None}")

    # Affichage des fichiers trouvés
    if os.path.exists('models'):
        files = os.listdir('models')
        logger.info(f"Fichiers dans models/: {files}")
    else:
        logger.warning("Répertoire models/ non trouvé")

    # Configuration selon l'environnement
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.environ.get('PORT', 5001))

    logger.info(f"Démarrage sur le port {port} (debug: {debug_mode})")

    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=port,
        threaded=True
    )