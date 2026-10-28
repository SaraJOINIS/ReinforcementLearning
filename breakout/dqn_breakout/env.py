import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
import ale_py

def make_env():
    gym.register_envs(ale_py)
    env = gym.make('ALE/Breakout-v5', obs_type='grayscale', render_mode=None, frameskip=8)
    env = ResizeObservation(env, shape=(84, 84))  
    return env
