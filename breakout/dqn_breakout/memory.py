import collections
import random
import numpy as np
import tensorflow as tf

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """
    Classe pour stocker et gérer la mémoire de replay utilisée dans l'entraînement DQN.
    Utilise une deque pour conserver un nombre fixe de transitions (expériences).
    """
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_tf(self, batch_size):
        transitions = self.sample(batch_size)
        batch = Transition(*zip(*transitions))

        states = tf.convert_to_tensor(np.array(batch.state), dtype=tf.float32)
        actions = tf.convert_to_tensor(batch.action, dtype=tf.int32)
        rewards = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(batch.next_state), dtype=tf.float32)
        dones = tf.convert_to_tensor(batch.done, dtype=tf.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
