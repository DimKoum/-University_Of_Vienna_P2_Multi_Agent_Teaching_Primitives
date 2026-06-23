import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_training_rewards(log_file_path, save_fig_name=None):
    df = pd.read_csv(log_file_path, skiprows=1)

    rewards = df["r"]
    timesteps = df["l"].cumsum()

    plt.plot(timesteps, rewards)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Training Rewards")
    plt.grid(True)

    if save_fig_name is not None:
        plt.savefig(f"{save_fig_name}.png")


def plot_elapsed_timesteps(timesteps_list, save_fig_name=None):
    x = np.arange(0, len(timesteps_list))
    y = timesteps_list

    plt.plot(x, y)

    plt.xlabel("Episode")
    plt.ylabel("Episode Elapsed Time(Timesteps)")
    plt.title("Elapsed Timesteps per episode")
    plt.grid(True)

    if save_fig_name is not None:
        plt.savefig(f"{save_fig_name}.png")


def plot_construction_ratios(correct_construction_ratios_list, save_fig_name=None):
    x = np.arange(0, len(correct_construction_ratios_list))
    y = correct_construction_ratios_list
    plt.plot(x, y)

    plt.xlabel("Episode")
    plt.ylabel("Correct Placement Ratio")
    plt.title("Correct Tile Placement Ratio per Episode")
    plt.grid(True)

    if save_fig_name is not None:
        plt.savefig(f"{save_fig_name}.png")


def line_Plot(x=None, y=None, xLabel="none", yLabel="none", title="none",save_fig_name = None):
    if x is None:
        x = [0]
    if y is None:
        y = [0]

    plt.plot(x, y)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid(True)

    if save_fig_name is not None:
        plt.savefig(f"{save_fig_name}.png")