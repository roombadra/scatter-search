from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'

# Fonction pour calculer la distance entre deux villes
def calculer_distance(ville1, ville2):
    return np.linalg.norm(ville1 - ville2)

# Fonction pour calculer la distance totale d'une solution TSP donnée
def calculer_distance_totale(solution, coordonnees_villes):
    distance_totale = 0
    num_villes = len(solution)

    for i in range(num_villes - 1):
        distance_totale += calculer_distance(coordonnees_villes[solution[i]], coordonnees_villes[solution[i + 1]])

    distance_totale += calculer_distance(coordonnees_villes[solution[-1]], coordonnees_villes[solution[0]])  # Retour à la ville de départ
    return distance_totale

# Fonction pour effectuer la diversification en échangeant au hasard deux villes dans une solution
def diversifier(solution):
    nouvelle_solution = solution.copy()
    idx1, idx2 = np.random.choice(len(solution), size=2, replace=False)
    nouvelle_solution[idx1], nouvelle_solution[idx2] = nouvelle_solution[idx2], nouvelle_solution[idx1]
    return nouvelle_solution

# Fonction pour sauvegarder le tracé du TSP
def save_tsp_plot(city_coordinates, solution, file_path):
    plt.figure(figsize=(8, 8))
    plt.plot(city_coordinates[:, 0], city_coordinates[:, 1], 'o', label='Villes')
    plt.plot(city_coordinates[solution, 0], city_coordinates[solution, 1], 'r-', linewidth=2, label='Chemin')
    plt.plot(city_coordinates[solution[0], 0], city_coordinates[solution[0], 1], 'go', label='Départ')
    plt.title('Solution du problème du voyageur de commerce')
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

# Algorithme de recherche éparse pour le problème du voyageur de commerce (TSP)
def recherche_eparse_tsp(coordonnees_villes, num_iterations, taille_ensemble_reference):
    num_villes = len(coordonnees_villes)
    
    # Générer un ensemble de référence initial avec des solutions aléatoires
    ensemble_reference = [np.random.permutation(num_villes) for _ in range(taille_ensemble_reference)]

    for iteration in range(num_iterations):
        # Phase de diversification : Appliquer une stratégie de diversification (par exemple, l'échange aléatoire de villes)
        nouvelles_solutions = [diversifier(solution) for solution in ensemble_reference]

        # Évaluer les nouvelles solutions
        distances_nouvelles_solutions = [calculer_distance_totale(sol, coordonnees_villes) for sol in nouvelles_solutions]

        # Mettre à jour l'ensemble de référence en sélectionnant des solutions diverses et de haute qualité
        solutions_combinees = ensemble_reference + nouvelles_solutions
        distances_combinees = [calculer_distance_totale(sol, coordonnees_villes) for sol in solutions_combinees]
        indices_tries = np.argsort(distances_combinees)
        ensemble_reference = [solutions_combinees[i] for i in indices_tries[:taille_ensemble_reference]]

        print(f"Itération {iteration + 1}: Meilleure distance - {distances_combinees[indices_tries[0]]}")

    # Retourner la meilleure solution trouvée
    meilleure_solution = ensemble_reference[0]
    meilleure_distance = calculer_distance_totale(meilleure_solution, coordonnees_villes)
    meilleure_solution = list(meilleure_solution)
    meilleure_solution.append(meilleure_solution[0])

    # Sauvegarde de l'image
    image_path = os.path.join(app.config['STATIC_FOLDER'], 'tsp_solution_plot.png')
    save_tsp_plot(coordonnees_villes, meilleure_solution, image_path)

    print(f"Meilleure solution trouvée : {meilleure_solution}")
    print(f"Distance totale : {meilleure_distance}")
    return meilleure_solution, image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    chemin_fichier = os.path.join(app.config['UPLOAD_FOLDER'], 'tiny.csv')  # Utilisation du chemin complet
    donnees_villes_df = pd.read_csv(chemin_fichier, header=None, names=['X', 'Y'])
    coordonnees_villes = donnees_villes_df.to_numpy()

    np.random.seed(42)
    taille_ensemble_reference = 5

    num_iterations = 100  # Valeur par défaut

    if request.method == 'POST':
        fichier = request.files['csvFile']
        if fichier:
            chemin_fichier = os.path.join(app.config['UPLOAD_FOLDER'], fichier.filename)
            fichier.save(chemin_fichier)
            donnees_villes_df = pd.read_csv(chemin_fichier, header=None, names=['X', 'Y'])
            coordonnees_villes = donnees_villes_df.to_numpy()

        num_iterations = int(request.form['numIterations'])  # Récupération du nombre d'itérations depuis le formulaire

    meilleure_solution, image_path = recherche_eparse_tsp(coordonnees_villes, num_iterations, taille_ensemble_reference)
    meilleure_distance = calculer_distance_totale(meilleure_solution, coordonnees_villes)

    return render_template('index.html', meilleure_solution=meilleure_solution, distance_totale=meilleure_distance, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
