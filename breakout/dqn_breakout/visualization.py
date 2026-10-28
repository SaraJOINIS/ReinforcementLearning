import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def plot_training_summary_with_video(reward_list, epsilon_list, frames, window=100):
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))  

    # Courbes comme avant
    axs[0].plot(reward_list, label="Récompense par épisode")
    axs[0].set_title("Récompenses brutes par épisode")
    axs[0].set_xlabel("Épisode")
    axs[0].set_ylabel("Récompense")
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

    moving_avg = [np.mean(reward_list[max(0, i-window):i+1]) for i in range(len(reward_list))]
    axs[1].plot(moving_avg, label=f"Moyenne mobile ({window})")
    axs[1].set_title("Récompense moyenne mobile")
    axs[1].set_xlabel("Épisode")
    axs[1].set_ylabel("Récompense")
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    axs[2].plot(epsilon_list, label="Epsilon")
    axs[2].set_title("Décroissance de l'epsilon")
    axs[2].set_xlabel("Épisode")
    axs[2].set_ylabel("Epsilon")
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend()

    # Animation vidéo dans axs[3]
    axs[3].axis('off')  # On cache les axes (optionnel)

    im = axs[3].imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

    plt.tight_layout()
    plt.show()