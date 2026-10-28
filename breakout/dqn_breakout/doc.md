# Détail des fichiers du projet

### `agent.py`

Contient des fonctions utilitaires pour l'entraînement d'un agent DQN :

- **Décroissance dynamique d’epsilon** (`get_epsilon_dynamic`) : ajuste la probabilité d’exploration au fil du temps selon une décroissance linéaire.
- **Sélection d’action epsilon-greedy** (`select_action`) : choisit une action aléatoire avec probabilité epsilon, sinon l’action optimale selon le réseau de politique.
- **Optimisation du modèle** (`optimize_model`) : réalise une mise à jour des poids du réseau de politique en utilisant un batch d’expériences échantillonné dans la mémoire, selon la méthode DQN classique.
- **Conversion des échantillons en tenseurs TensorFlow** (`sample_tf`) : transforme un batch d’expériences en tenseurs adaptés à TensorFlow pour l’entraînement.

---

### `config.py`

Centralise tous les paramètres et hyperparamètres utilisés :

* `learning_rate`, `gamma`, `epsilon`, etc.

---

### `env.py`

Jeu a atari Breakout ALE/Breakout-v5, noir est blanc, format  84 X 84

---

### `main.py`

Script principal d’exécution :

* Lance l'entraînement et/ou les tests
* Gère la boucle principale d’interaction **agent ↔ environnement**

---

### `memory.py`

Gère la mémoire d’expérience (replay buffer) :

- Stocke les transitions (état, action, récompense, état suivant, done) dans une structure fixe (deque).
- Permet d’ajouter de nouvelles expériences (push).
- Offre un échantillonnage aléatoire (sample) pour casser la corrélation entre expériences.
- Fournit une conversion des échantillons en tenseurs TensorFlow pour l’entraînement (sample_tf).

---

### `model.py`

Construit et compile un **réseau de neurones convolutionnel** (CNN) pour approximer la fonction Q dans le cadre du DQN :

- **Entrée** : une image représentant l’état (ex. : 4 dernières frames du jeu).
- **Convolutions + Pooling** : extraient et réduisent les caractéristiques visuelles importantes.
- **Flatten + Dense** : convertissent les features extraits en valeurs Q pour chaque action possible.
- **Compilation** : utilise l'optimiseur Adam et la perte MSE pour entraîner le modèle à prédire les bonnes Q-values.
---

### `preprocess.py`

Effectue le **prétraitement** des observations :

Stocke et empile les dernières **k** images (frames) observées par l’agent.

Permet d’ajouter un contexte temporel aux observations, utile pour les algorithmes d’apprentissage par renforcement.

Transforme l’observation pour gérer les images en niveaux de gris ou en couleur.

Modifie l’espace d’observation pour refléter la concaténation des **k** frames.

Ce code définit une classe **FrameStack** qui est un wrapper pour un environnement Gymnasium. Cette classe permet de stocker et empiler les dernières **k** images (frames) observées par l’agent, ce qui est utile pour donner un contexte temporel aux algorithmes d’apprentissage par renforcement.

---

### `visualization.py`

Permet la **visualisation** des performances de l’agent :

* Graphiques de récompenses par épisode
* Suivi de l’exploration, des pertes, etc.
* Fonction de synthèse graphique et vidéo via `plot_training_summary_with_video()` :

  * Affiche :

    * Récompenses brutes
    * Moyenne mobile des récompenses
    * Évolution d’epsilon
    * Animation d’une séquence de frames (vidéo du comportement de l’agent)

---