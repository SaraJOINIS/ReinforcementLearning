import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange
from env import make_env
from memory import ReplayMemory
from preprocess import FrameStack  
from model import build_model
from agent import get_epsilon_dynamic, select_action, optimize_model
from visualization import plot_training_summary_with_video
from config import *

# === Initialisation environnement, modèle et mémoire ===
env = make_env()
env = FrameStack(env, STACK_SIZE)
n_actions = env.action_space.n

memory = ReplayMemory(MEMORY_SIZE)
input_shape = env.observation_space.shape
policy_net = build_model(input_shape, n_actions, LR)
target_net = build_model(input_shape, n_actions, LR)
target_net.set_weights(policy_net.get_weights())

frame_idx = 0
reward_list = []
epsilon_list = []
frames_last_episode = []
average_frame_by_episode = []

for episode in trange(NUM_EPISODES, desc="Training"):
    state, info = env.reset()
    lives = info.get("lives", 0)
    done = False
    total_reward = 0
    frame_in_ep = 0

    if episode == NUM_EPISODES - 1:
        frames_last_episode = [state]
    else:
        frames_last_episode = None

    while not done and frame_in_ep < MAX_FRAMES_PER_EPISODE:
        epsilon = get_epsilon_dynamic(frame_idx, EPS_START, EPS_END, TOTAL_FRAMES_TO_DECAY)
        action = select_action(state, epsilon, policy_net, n_actions)
        next_state, reward, terminated, truncated, next_info = env.step(action)

        new_lives = next_info.get("lives", lives)
        lost_life = new_lives < lives and new_lives > 0  # Vie perdue mais partie non finie
        done = terminated or truncated or lost_life

        memory.push(state, action, reward, next_state, done)

        lives = new_lives  # Met à jour des vies restantes
        state = next_state
        total_reward += reward
        frame_idx += 1
        frame_in_ep += 1

        if frames_last_episode is not None:
            frames_last_episode.append(next_state)

        if len(memory) >= BATCH_SIZE:
            optimize_model(memory, policy_net, target_net, BATCH_SIZE, GAMMA)

        if frame_idx % TARGET_UPDATE == 0:
            target_net.set_weights(policy_net.get_weights())

    reward_list.append(total_reward)
    epsilon_list.append(epsilon)
    average_frame_by_episode.append(frame_in_ep)

    avg_frames = sum(average_frame_by_episode) / len(average_frame_by_episode)

    print(f"Episode {episode+1}/{NUM_EPISODES} - Reward: {total_reward:.1f} - Epsilon: {epsilon:.3f} - Frames: {frame_in_ep} - Moyenne Frames/Episode: {avg_frames:.1f}")

    if (episode + 1) % 50 == 0:
        policy_net.save(f"models/policy_net_ep{episode+1}.h5")

env.close()

