import joblib
import json


def fix_mappings():
    print("Correction des noms de colonnes dans les mappings...")

    # Charger les mappings existants
    mappings = joblib.load('models/label_mappings.pkl')

    # Créer les nouveaux mappings avec les bons noms
    new_mappings = {}

    # Mapping des anciens noms vers les nouveaux
    column_rename = {
        'Filiere': 'Filiere',
        'aff_dom': 'aff_dom',
        'moy_sci': 'moy_sci',
        'moy_ann': 'moy_ann',
        'sect_parent': 'sect_parent',
        '6- Dans quel domaine d\'activité avez-vous une fois participé ou mené une activités? ': 'part_act',
        '7- A quel secteur d\'activité, l\'accessibilité au marché de l\'emploi est le plus offrant ? ': 'acc_empl'
    }

    # Renommer les clés
    for old_name, new_name in column_rename.items():
        if old_name in mappings:
            new_mappings[new_name] = mappings[old_name]
            print(f"✅ {old_name} -> {new_name}")
        else:
            print(f"❌ {old_name} non trouvé dans les mappings")

    # Vérifier que toutes les variables attendues sont présentes
    expected_vars = ['aff_dom', 'moy_sci', 'moy_ann', 'sect_parent', 'part_act', 'acc_empl']
    for var in expected_vars:
        if var in new_mappings:
            print(f"✅ {var}: {list(new_mappings[var].keys())}")
        else:
            print(f"❌ {var}: MANQUANT")

    # Sauvegarder les nouveaux mappings
    joblib.dump(new_mappings, 'models/label_mappings.pkl')
    print("\n✅ Mappings corrigés et sauvegardés")

    return new_mappings


if __name__ == "__main__":
    mappings = fix_mappings()