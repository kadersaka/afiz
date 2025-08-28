# Documentation Système de Recommandation Académique

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du Système](#architecture-du-système)
3. [Installation et Configuration](#installation-et-configuration)
4. [Backend Flask](#backend-flask)
5. [Modèles Machine Learning](#modèles-machine-learning)
6. [API Documentation](#api-documentation)
7. [Frontend Web](#frontend-web)
8. [Déploiement](#déploiement)
9. [Tests et Validation](#tests-et-validation)
10. [Maintenance](#maintenance)
11. [Dépannage](#dépannage)

## Vue d'ensemble

### Objectif du Projet

Le Système de Recommandation Académique est une application web intelligente conçue pour orienter les étudiants vers la filière d'études la plus adaptée à leur profil. Le système analyse 6 critères clés pour recommander l'une des trois filières disponibles :

- **Informatique**
- **Médecine** 
- **Agronomie**

### Technologies Utilisées

**Backend :**
- Python 3.8+
- Flask (API REST)
- Scikit-learn (Machine Learning)
- Pandas (Manipulation de données)
- NumPy (Calculs numériques)

**Frontend :**
- HTML5
- CSS3 (avec animations et responsive design)
- JavaScript Vanilla (ES6+)

**Machine Learning :**
- K-Nearest Neighbors (KNN)
- Random Forest
- Label Encoding pour variables catégorielles

## Architecture du Système

```
┌─────────────────┐    HTTP/JSON    ┌─────────────────┐
│                 │  ────────────>  │                 │
│   Frontend Web  │                 │  Backend Flask  │
│   (HTML/CSS/JS) │  <────────────  │     (API)       │
│                 │                 │                 │
└─────────────────┘                 └─────────────────┘
                                            │
                                            │
                                    ┌───────▼────────┐
                                    │  Modèles ML    │
                                    │  - KNN Model   │
                                    │  - RF Model    │
                                    │  - Mappings    │
                                    └────────────────┘
```

### Structure des Fichiers

```
projet/
├── README.md                    # Cette documentation
├── app.py                      # Application Flask principale
├── ml_predictor.py            # Moteur de recommandation
├── data_transformer.py        # Transformation et validation
├── train_models.py           # Entraînement des modèles
├── test_models.py           # Tests unitaires
├── index.html              # Interface web frontend
├── models/                # Modèles sauvegardés
│   ├── knn_model.pkl
│   ├── rf_model.pkl
│   ├── label_mappings.pkl
│   └── model_metadata.json
├── data/                 # Données d'entraînement
│   └── quest_data.xlsx
└── requirements.txt     # Dépendances Python
```

## Installation et Configuration

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Navigateur web moderne

### Installation

1. **Cloner ou télécharger le projet**
```bash
# Si vous utilisez git
git clone <repository-url>
cd academic-recommendation-system

# Ou créer le dossier manuellement et y placer les fichiers
```

2. **Créer un environnement virtuel**
```bash
python -m venv .venv

# Activation
# Sur macOS/Linux
source .venv/bin/activate

# Sur Windows
.venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install flask flask-cors pandas numpy scikit-learn joblib openpyxl
```

4. **Créer le fichier requirements.txt**
```txt
Flask==2.3.3
Flask-CORS==4.0.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
openpyxl==3.1.2
```

### Configuration initiale

1. **Placer le fichier de données**
   - Assurez-vous que `quest_data.xlsx` est dans le dossier `data/`

2. **Entraîner les modèles**
```bash
python train_models.py
```

3. **Vérifier l'installation**
```bash
python test_models.py
```

## Backend Flask

### Structure de l'Application

L'application Flask est organisée en modules :

#### app.py - Application Principale
- Configuration Flask et CORS
- Définition des routes API
- Gestion d'erreurs globale
- Initialisation des modèles ML

#### ml_predictor.py - Moteur de Recommandation
```python
class RecommendationEngine:
    def __init__(self, models_dir="models")
    def get_recommendation(self, user_responses)
    def validate_user_input(self, user_data)
    def get_model_info(self)
```

#### data_transformer.py - Transformation des Données
```python
class DataTransformer:
    def get_questions(self)
    def validate_user_input(self, form_data)
    def transform_categorical_to_numerical(self, responses)
    def get_form_schema(self)
```

### Configuration du Serveur

**Démarrage standard :**
```bash
python app.py
```

**Démarrage sur un port spécifique :**
```bash
python app.py --port 5001
```

**Configuration via variables d'environnement :**
```bash
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5001
export FLASK_DEBUG=True
python app.py
```

### Variables de Configuration

```python
class Config:
    HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
    PORT = int(os.environ.get('FLASK_PORT', 5001))
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
```

## Modèles Machine Learning

### Données d'Entraînement

**Source :** 160 observations d'étudiants avec leurs réponses et filières choisies

**Variables d'entrée (6) :**
1. `aff_dom` - Affinités avec les domaines
2. `moy_sci` - Moyennes en sciences  
3. `moy_ann` - Moyennes annuelles
4. `sect_parent` - Secteur professionnel des proches
5. `part_act` - Participation à des activités
6. `acc_empl` - Perception de l'employabilité

**Variable cible :** Filière choisie (Informatique, Médecine, Agronomie)

### Algorithmes Utilisés

#### K-Nearest Neighbors (KNN)
```python
KNeighborsClassifier(
    n_neighbors=10,
    metric='manhattan'
)
```
- **Précision sur test :** 75.0%
- **Avantages :** Simple, interprétable, performant pour ce type de données
- **Inconvénients :** Sensible aux données déséquilibrées

#### Random Forest
```python
RandomForestClassifier(
    max_depth=5,
    n_estimators=100,
    random_state=42
)
```
- **Précision sur test :** 78.1%
- **Avantages :** Robuste, gère bien la non-linéarité
- **Inconvénients :** Moins interprétable

### Pipeline de Traitement

1. **Chargement des données** depuis Excel
2. **Renommage des colonnes** pour faciliter la manipulation
3. **Encodage des variables catégorielles** avec LabelEncoder
4. **Division train/test** (80/20, random_state=42)
5. **Optimisation des hyperparamètres** avec GridSearchCV
6. **Entraînement des modèles** finaux
7. **Sauvegarde** des modèles et métadonnées

### Logique de Consensus

Le système utilise les deux modèles et applique cette logique :

```python
if knn_prediction == rf_prediction:
    final_recommendation = knn_prediction
    confidence = 0.9  # Haute confiance
else:
    # Utiliser le modèle le plus performant
    if rf_accuracy > knn_accuracy:
        final_recommendation = rf_prediction
        confidence = 0.7
    else:
        final_recommendation = knn_prediction
        confidence = 0.7
```

## API Documentation

### Base URL
```
http://127.0.0.1:5001
```

### Endpoints

#### 1. Informations Générales
```http
GET /
```

**Réponse :**
```json
{
  "message": "API de Recommandation Académique",
  "version": "1.0.0",
  "status": "active",
  "models_loaded": true,
  "endpoints": {...}
}
```

#### 2. Health Check
```http
GET /api/health
```

**Réponse (200 - Healthy) :**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-28T11:38:30.191911",
  "models_status": {
    "knn_loaded": true,
    "rf_loaded": true,
    "mappings_loaded": true
  },
  "system_info": {
    "uptime_seconds": 3600,
    "total_predictions": 42
  }
}
```

#### 3. Structure du Formulaire
```http
GET /api/questions
```

**Réponse :**
```json
{
  "success": true,
  "questions": {
    "aff_dom": {
      "question": "Avec quel domaine avez-vous des affinités ?",
      "type": "select",
      "required": true,
      "options": [
        "Domaine Industriel",
        "Domaine de l'agronomie", 
        "Domaine de la santé"
      ],
      "help_text": "Choisissez le domaine qui vous attire..."
    }
  },
  "total_questions": 6
}
```

#### 4. Validation des Données
```http
POST /api/validate
Content-Type: application/json
```

**Body :**
```json
{
  "aff_dom": "Domaine Industriel",
  "moy_sci": "Entre 10/20 et 15/20",
  "moy_ann": "Entre 10/20 et 15/20",
  "sect_parent": "Secteur de la santé",
  "part_act": "Domaine industriel",
  "acc_empl": "Secteur industriel"
}
```

**Réponse :**
```json
{
  "success": true,
  "validation": {
    "is_valid": true,
    "errors": [],
    "cleaned_data": {...}
  }
}
```

#### 5. Recommandation (Principal)
```http
POST /api/recommend
Content-Type: application/json
```

**Body :** Même format que la validation

**Réponse :**
```json
{
  "success": true,
  "recommendation": {
    "final_recommendation": "Informatique",
    "confidence": 0.9,
    "knn_prediction": "Informatique",
    "rf_prediction": "Informatique", 
    "model_agreement": true,
    "explanation": "Les deux modèles s'accordent...",
    "probabilities": {
      "Informatique": 0.85,
      "Médecine": 0.10,
      "Agronomie": 0.05
    },
    "detailed_probabilities": {
      "knn": {...},
      "rf": {...}
    }
  },
  "processing_time_ms": 22.44,
  "timestamp": "2025-08-28T11:38:30.191911"
}
```

#### 6. Informations sur les Modèles
```http
GET /api/models/info
```

**Réponse :**
```json
{
  "success": true,
  "models_info": {
    "models_loaded": true,
    "knn_accuracy": 0.750,
    "rf_accuracy": 0.781,
    "features": [
      "aff_dom", "moy_sci", "moy_ann", 
      "sect_parent", "part_act", "acc_empl"
    ],
    "target_classes": [
      "Agronomie", "Informatique", "Médecine"
    ]
  }
}
```

### Gestion d'Erreurs

#### Erreur 400 - Données Invalides
```json
{
  "success": false,
  "error": "Données invalides",
  "details": [
    "Variable manquante: moy_ann",
    "Valeur invalide 'Domaine Inexistant' pour aff_dom"
  ],
  "timestamp": "2025-08-28T11:38:30.191911"
}
```

#### Erreur 503 - Service Non Disponible
```json
{
  "success": false,
  "error": "Modèles non disponibles",
  "message": "Exécutez 'python train_models.py'",
  "timestamp": "2025-08-28T11:38:30.191911"
}
```

## Frontend Web

### Structure HTML

Le frontend est une single-page application avec :

```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Système de Recommandation Académique</title>
    <!-- Styles CSS intégrés -->
</head>
<body>
    <div class="container">
        <!-- Header -->
        <!-- Formulaire -->
        <!-- Loading -->
        <!-- Résultats -->
        <!-- Gestion d'erreurs -->
    </div>
    <!-- JavaScript intégré -->
</body>
</html>
```

### Fonctionnalités Frontend

#### Interface Utilisateur
- **Design responsive** : S'adapte mobile et desktop
- **Barre de progression** : Suivi visuel du remplissage
- **Animations CSS** : Transitions fluides
- **Thème moderne** : Dégradés et ombres

#### Interactions
- **Validation temps réel** : Vérification au fur et à mesure
- **États de chargement** : Spinner pendant les requêtes
- **Gestion d'erreurs** : Messages contextuels
- **Restart facile** : Bouton pour recommencer

#### Affichage des Résultats
```javascript
function displayResults(recommendation) {
    // Badge coloré selon la filière
    const badge = document.getElementById('recommendationBadge');
    badge.className = `badge-${recommendation.final_recommendation.toLowerCase()}`;
    
    // Barre de confiance
    const confidence = Math.round(recommendation.confidence * 100);
    document.getElementById('confidenceFill').style.width = confidence + '%';
    
    // Probabilités avec barres visuelles
    displayProbabilities(recommendation.probabilities);
}
```

### Configuration API

```javascript
const API_BASE_URL = 'http://127.0.0.1:5001';

// Test de connexion au démarrage
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (!response.ok) {
            console.warn('API Flask non disponible');
        }
    } catch (error) {
        console.warn('Impossible de se connecter à l\'API');
    }
});
```

## Déploiement

### Développement Local

1. **Backend :**
```bash
python app.py --port 5001
```

2. **Frontend :**
   - Ouvrir `index.html` dans un navigateur
   - Ou utiliser un serveur local : `python -m http.server 8000`

### Production

#### Backend Flask avec Gunicorn
```bash
# Installation
pip install gunicorn

# Lancement
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

#### Configuration Nginx (optionnel)
```nginx
server {
    listen 80;
    server_name votre-domaine.com;
    
    location / {
        root /path/to/frontend;
        index index.html;
    }
    
    location /api/ {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Variables d'Environnement Production
```bash
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5001
export FLASK_DEBUG=False
export MODELS_DIR=/path/to/models
```

### Docker (optionnel)

**Dockerfile :**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "app.py", "--port", "5001", "--host", "0.0.0.0"]
```

**docker-compose.yml :**
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "5001:5001"
    volumes:
      - "./models:/app/models"
      - "./data:/app/data"
    environment:
      - FLASK_HOST=0.0.0.0
      - FLASK_PORT=5001
```

## Tests et Validation

### Tests Unitaires

**Exécution :**
```bash
python test_models.py
```

**Couverture :**
- Chargement des modèles
- Validation des données
- Précision des prédictions
- Transformation des variables
- Gestion d'erreurs

### Tests API

**Avec curl :**
```bash
# Health check
curl http://127.0.0.1:5001/api/health

# Recommandation
curl -X POST http://127.0.0.1:5001/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"aff_dom":"Domaine Industriel",...}'
```

**Avec Postman :**
- Importer la collection depuis la documentation API
- Configurer `base_url = http://127.0.0.1:5001`
- Tester tous les endpoints

### Profils de Test

1. **Profil Informatique :**
```json
{
  "aff_dom": "Domaine Industriel",
  "moy_sci": "Entre 10/20 et 15/20",
  "moy_ann": "Entre 10/20 et 15/20",
  "sect_parent": "Secteur industriel",
  "part_act": "Domaine industriel",
  "acc_empl": "Secteur industriel"
}
```

2. **Profil Médecine :**
```json
{
  "aff_dom": "Domaine de la santé",
  "moy_sci": ">15/20",
  "moy_ann": "Entre 10/20 et 15/20",
  "sect_parent": "Secteur de la santé",
  "part_act": "Domaine de la santé",
  "acc_empl": "Secteur de la santé"
}
```

3. **Profil Agronomie :**
```json
{
  "aff_dom": "Domaine de l'agronomie",
  "moy_sci": "Entre 10/20 et 15/20",
  "moy_ann": "Entre 10/20 et 15/20",
  "sect_parent": "Secteur agricol",
  "part_act": "Domaine agricol",
  "acc_empl": "Secteur agricol"
}
```

## Maintenance

### Logs et Monitoring

**Configuration du logging :**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

**Métriques à surveiller :**
- Temps de réponse API
- Taux d'erreur
- Distribution des recommandations
- Utilisation mémoire/CPU

### Mise à Jour des Modèles

1. **Collecte de nouvelles données**
2. **Ajout au dataset existant**
3. **Réentraînement :**
```bash
python train_models.py
```
4. **Tests de validation**
5. **Redémarrage de l'API**

### Backup

**Éléments à sauvegarder :**
- Fichiers de modèles (`models/`)
- Données d'entraînement (`data/`)
- Logs d'application
- Configuration

## Dépannage

### Problèmes Courants

#### 1. "Mapping manquant pour la variable: part_act"

**Cause :** Mappings de labels incomplets ou corrompus

**Solution :**
```bash
python -c "
import joblib
mappings = joblib.load('models/label_mappings.pkl')
print('Variables:', list(mappings.keys()))
"

# Si part_act manque, régénérer :
python train_models.py
```

#### 2. "Modèles non disponibles"

**Cause :** Fichiers de modèles manquants

**Solution :**
```bash
ls -la models/
# Vérifier la présence de :
# - knn_model.pkl
# - rf_model.pkl  
# - label_mappings.pkl
# - model_metadata.json

# Si manquants :
python train_models.py
```

#### 3. "Impossible de se connecter au serveur"

**Cause :** API Flask non démarrée ou port incorrect

**Solution :**
```bash
# Vérifier que l'API tourne
curl http://127.0.0.1:5001/api/health

# Relancer l'API
python app.py --port 5001

# Vérifier les ports utilisés
netstat -an | grep 5001
```

#### 4. Erreurs de validation côté frontend

**Cause :** Valeurs non conformes aux options attendues

**Solution :**
1. Vérifier les options disponibles : `GET /api/questions`
2. S'assurer que les valeurs du frontend correspondent exactement
3. Attention aux accents et espaces

#### 5. Performance dégradée

**Causes possibles :**
- Modèles trop volumineux
- Manque de mémoire
- Trop de requêtes simultanées

**Solutions :**
- Optimiser les modèles
- Augmenter la mémoire disponible  
- Implémenter du caching
- Utiliser un load balancer

### Scripts de Diagnostic

**Diagnostic complet :**
```bash
python -c "
import os
import joblib
from pathlib import Path

print('=== DIAGNOSTIC SYSTÈME ===')
print(f'Répertoire: {os.getcwd()}')
print(f'Fichiers Python: {[f for f in os.listdir('.') if f.endswith('.py')]}')
print(f'Répertoire models: {os.path.exists('models')}')

if os.path.exists('models'):
    files = os.listdir('models')
    print(f'Fichiers models: {files}')
    
    try:
        mappings = joblib.load('models/label_mappings.pkl')
        print(f'Variables mappings: {list(mappings.keys())}')
    except Exception as e:
        print(f'Erreur mappings: {e}')
"
```

### Support et Contact

Pour les problèmes techniques non couverts par cette documentation :

1. Vérifier les logs d'application
2. Reproduire le problème avec les données de test
3. Consulter la section dépannage appropriée
4. Documenter l'erreur avec les logs pertinents

---

*Documentation générée pour le Système de Recommandation Académique - Version 1.0*