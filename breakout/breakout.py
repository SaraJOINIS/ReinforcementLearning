import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# Hyperparamètres
gamma = 0.99            # Facteur de réduction des récompenses futures
epsilon = 1.0           # Paramètre epsilon pour la stratégie epsilon-greedy
epsilon_min = 0.1       # Epsilon minimum
epsilon_decay = 0.995   # Décroissance d'epsilon
batch_size = 32         # Taille du batch pour la mise à jour du modèle
memory = []             # Mémoire pour l'expérience replay
max_memory = 100000     # Taille maximale de la mémoire
learning_rate = 0.00025 # Taux d'apprentissage du modèle
update_target_every = 10000  # Nombre de pas avant de mettre à jour le modèle cible


env = gym.make('ALE/Breakout-v5', obs_type="grayscale", render_mode="human")
obs, info = env.reset()

# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

for _ in range(200):
    # Choisir une action aléatoire
    action = env.action_space.sample()

    # Effectuer l'action dans l'environnement
    observation, reward, terminated, truncated, info = env.step(action)

    # (Pas de env.render())

    # Vérifier si la partie est terminée
    if terminated or truncated:
        observation, info = env.reset()

# Fermer proprement
env.close()
