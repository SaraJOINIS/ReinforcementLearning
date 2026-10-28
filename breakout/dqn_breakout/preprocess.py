import gymnasium as gym
import numpy as np
from collections import deque
class FrameStack(gym.Wrapper):
    """
    Un wrapper Gymnasium qui empile les dernières k observations (frames) 
    de l’environnement pour fournir un contexte temporel à l’agent.

    Cela permet aux algorithmes de deep reinforcement learning de mieux percevoir 
    le mouvement dans les environnements visuels (comme les jeux Atari).
    """
    
    def __init__(self, env, k):
        """
        Initialise le FrameStack.

        Args:
            env (gym.Env): Environnement Gymnasium à wrapper.
            k (int): Nombre de frames à empiler.
        """
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)

        shp = env.observation_space.shape
        # Gère les observations en niveaux de gris en leur ajoutant une dimension de canal
        if len(shp) == 2:
            h, w = shp
            c = 1
        else:
            h, w, c = shp

        # Redéfinit l’espace d’observation pour inclure k frames empilées
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c * k),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        """
        Réinitialise l’environnement et empile k copies de la première observation 
        pour initialiser l’historique.

        Returns:
            observation empilée, info dictionnaire
        """
        obs, info = self.env.reset(**kwargs)
        if obs.ndim == 2:  # observation en niveaux de gris
            obs = obs[..., np.newaxis]

        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        """
        Applique une action à l’environnement et met à jour la pile de frames.

        Args:
            action (int): Action choisie par l’agent

        Returns:
            observation empilée, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if obs.ndim == 2:
            obs = obs[..., np.newaxis]
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """
        Concatène les k dernières observations le long de l’axe des canaux.

        Returns:
            np.ndarray: Observation empilée sur les k dernières frames
        """
        return np.concatenate(list(self.frames), axis=2)
