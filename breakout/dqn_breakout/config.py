# === Hyperparamètres ===
GAMMA = 0.99                # Facteur de discount classique
EPS_START = 1.0             # Exploration totale au début
EPS_END = 0.05               # Exploration minimale en fin de décroissance
BATCH_SIZE = 64             # Taille de batch raisonnable pour stabilité et vitesse
LR = 0.0001                # Learning rate bas pour convergence stable
MEMORY_SIZE = 500000        # Mémoire importante pour diversifier les expériences
TARGET_UPDATE = 10000       # Mise à jour de la target toutes les 10k frames
STACK_SIZE = 4              # Empilement de frames classique pour Atari-like
NUM_EPISODES = 10000          # Nombre d'épisodes pour un entraînement correct

MAX_FRAMES_PER_EPISODE = 10000  # Limite haute pour éviter épisodes trop longs
TOTAL_FRAMES_TO_DECAY = 1_000_000  # epsilon décroit sur 2 millions de frames (~200 épisodes à 10k frames)
