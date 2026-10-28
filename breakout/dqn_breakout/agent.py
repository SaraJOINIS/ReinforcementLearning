import numpy as np
import random
import collections
import tensorflow as tf


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def get_epsilon_dynamic(frame_idx, EPS_START, EPS_END, total_frames_to_decay):
    """
    Calcule la valeur d'epsilon (exploration) à un instant donné en fonction
    du nombre de frames jouées pour une décroissance linéaire.
    
    Args:
        frame_idx (int): nombre total de frames jouées jusqu'à présent
        total_frames_to_decay (int): nombre total de frames sur lesquelles epsilon diminue
    
    Returns:
        float: valeur d'epsilon entre EPS_START et EPS_END
    """
    if frame_idx >= total_frames_to_decay:
        return EPS_END
    else:
        epsilon = EPS_START - (EPS_START - EPS_END) * (frame_idx / total_frames_to_decay)
        return epsilon

def select_action(state, epsilon, policy_net, n_actions):
    """
    Sélectionne une action selon la politique epsilon-greedy.
    
    Args:
        state (np.array): état actuel (stack de frames)
        epsilon (float): probabilité d'explorer (choisir une action aléatoire)
    
    Returns:
        int: action choisie
    """
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        state_t = np.expand_dims(state, axis=0)
        q_values = policy_net.predict(state_t, verbose=0)
        return np.argmax(q_values[0])


# Le décorateur @tf.function convertit la fonction Python en graphe TensorFlow optimisé.
# Cela permet une exécution plus rapide en compilant la fonction en opérations TensorFlow,
# tout en conservant une syntaxe Python classique. Utile pour améliorer les performances
# lors de l'entraînement du modèle.
@tf.function
@tf.function
def optimize_model(memory, policy_net, target_net, batch_size, gamma):
    """
    Met à jour les poids du réseau de politique (policy_net) en utilisant un batch d'expériences
    provenant de la mémoire de replay, selon l'algorithme DQN.

    Args:
        memory: objet ReplayMemory contenant les expériences passées
        policy_net: modèle principal utilisé pour prédire les Q-values
        target_net: modèle cible utilisé pour calculer les Q-values de l'état suivant
        batch_size (int): nombre d’échantillons utilisés à chaque mise à jour
        gamma (float): facteur de réduction du futur (discount factor)
    """
    
    # Échantillonnage du batch
    states, actions, rewards, next_states, dones = memory.sample_tf(batch_size)

    # Prédiction des Q-values actuelles et futures
    q_values = policy_net(states)
    q_next = target_net(next_states)
    max_q_next = tf.reduce_max(q_next, axis=1)

    # Calcul des cibles Q avec l'équation de Bellman
    target_q = rewards + gamma * max_q_next * (1.0 - dones)

    # Création du mask pour ne garder que la Q-value de l'action réellement choisie
    actions_one_hot = tf.one_hot(actions, policy_net.output_shape[-1])
    selected_q = tf.reduce_sum(q_values * actions_one_hot, axis=1)

    # Backpropagation : calcul du loss et mise à jour des poids du réseau de politique
    with tf.GradientTape() as tape:
        q_values = policy_net(states, training=True)
        selected_q = tf.reduce_sum(q_values * actions_one_hot, axis=1)
        loss = tf.reduce_mean(tf.square(target_q - selected_q))

    grads = tape.gradient(loss, policy_net.trainable_variables)
    policy_net.optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))


def sample_tf(self, batch_size):
    """
    Échantillonne un batch aléatoire d’expériences depuis la mémoire et
    les convertit en tenseurs TensorFlow pour l'entraînement.

    Args:
        batch_size (int): nombre de transitions à échantillonner

    Returns:
        Tuple de tenseurs (states, actions, rewards, next_states, dones)
    """
    transitions = random.sample(self.memory, batch_size)
    batch = Transition(*zip(*transitions))

    states = tf.convert_to_tensor(np.array(batch.state), dtype=tf.float32)
    actions = tf.convert_to_tensor(batch.action, dtype=tf.int32)
    rewards = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array(batch.next_state), dtype=tf.float32)
    dones = tf.convert_to_tensor(batch.done, dtype=tf.float32)

    return states, actions, rewards, next_states, dones

