# recreate_mappings.py
import pandas as pd
import joblib
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def recreate_complete_mappings():
    print("Régénération complète des mappings...")

    # Charger le fichier Excel
    data = pd.read_excel('data/quest_data.xlsx')
    print(f"Données chargées: {data.shape}")
    print(f"Colonnes originales: {list(data.columns)}")

    # Renommage explicite des colonnes
    data.columns = [
        'Filiere',  # Colonne 1
        'aff_dom',  # Colonne 2
        'moy_sci',  # Colonne 3
        'moy_ann',  # Colonne 4
        'sect_parent',  # Colonne 5
        'part_act',  # Colonne 6
        'acc_empl'  # Colonne 7
    ]

    print(f"Colonnes renommées: {list(data.columns)}")

    # Vérifier qu'on a bien 7 colonnes
    if len(data.columns) != 7:
        print(f"ERREUR: Attendu 7 colonnes, trouvé {len(data.columns)}")
        return False

    # Afficher quelques exemples de données
    print("\nAperçu des données:")
    print(data.head(3))

    # Créer les mappings pour toutes les variables
    label_mappings = {}
    encoded_data = pd.DataFrame()

    for col in data.columns:
        print(f"\nTraitement de {col}:")

        # Valeurs uniques
        unique_values = data[col].unique()
        print(f"  Valeurs uniques: {list(unique_values)}")

        # Encodage
        le = LabelEncoder()
        encoded_values = le.fit_transform(data[col])

        # Créer le mapping
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        label_mappings[col] = mapping
        encoded_data[col] = encoded_values

        print(f"  Mapping: {mapping}")

    print(f"\nMappings créés pour: {list(label_mappings.keys())}")

    # Sauvegarder les mappings
    joblib.dump(label_mappings, 'models/label_mappings.pkl')

    # Régénérer rapidement les modèles
    X = encoded_data.drop("Filiere", axis=1)
    y = encoded_data["Filiere"]

    print(f"\nDonnées d'entraînement: {X.shape}")
    print(f"Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner les modèles
    knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
    rf = RandomForestClassifier(max_depth=5, random_state=42)

    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    knn_score = knn.score(X_test, y_test)
    rf_score = rf.score(X_test, y_test)

    print(f"KNN accuracy: {knn_score:.4f}")
    print(f"RF accuracy: {rf_score:.4f}")

    # Sauvegarder les modèles
    joblib.dump(knn, 'models/knn_model.pkl')
    joblib.dump(rf, 'models/rf_model.pkl')

    # Métadonnées
    metadata = {
        'models': {
            'knn': {'performance': {'accuracy': float(knn_score)}},
            'rf': {'performance': {'accuracy': float(rf_score)}}
        },
        'feature_names': list(X.columns),
        'target_mapping': {0: 'Agronomie', 1: 'Informatique', 2: 'Médecine'}
    }

    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Régénération terminée!")
    return True


def verify_mappings():
    print("\nVérification des mappings régénérés:")

    mappings = joblib.load('models/label_mappings.pkl')
    expected_features = ['aff_dom', 'moy_sci', 'moy_ann', 'sect_parent', 'part_act', 'acc_empl']

    print(f"Variables trouvées: {list(mappings.keys())}")

    for feature in expected_features:
        if feature in mappings:
            print(f"✓ {feature}: {list(mappings[feature].keys())}")
        else:
            print(f"✗ {feature}: MANQUANT")

    return all(feature in mappings for feature in expected_features)


if __name__ == "__main__":
    success = recreate_complete_mappings()
    if success:
        if verify_mappings():
            print("\n✓ Tous les mappings sont maintenant présents!")
            print("Redémarrez votre application Flask")
        else:
            print("\n✗ Certains mappings sont encore manquants")
    else:
        print("Échec de la régénération")