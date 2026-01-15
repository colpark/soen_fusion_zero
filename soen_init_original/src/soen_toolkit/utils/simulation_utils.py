# FILEPATH: src/soen_toolkit/utils/simulation_utils.py

import matplotlib.pyplot as plt
import numpy as np
import torch


def clamp_tensor(tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clamp the given tensor element-wise between [min_val, max_val]."""
    return torch.clamp(tensor, min_val, max_val)


def plot_state_trajectory(states, node_indices=None, title="", show_legend=False, dpi=200, figsize=(10, 5), colors=None, input_flux=None):
    """Plot the states of specific nodes over time, with an optional thin plot of input flux at the top.

    Args:
        states (torch.Tensor or np.ndarray): Shape [batch_size, time_steps, n_nodes].
        node_indices (list, optional): List of node indices to plot. Defaults to all nodes.
        title (str): Title for the plot.
        show_legend (bool): Whether to show the legend.
        dpi (int): Dots per inch for the plot.
        figsize (tuple): Figure size for the plot.
        colors (list): List of colors for each node plot.
        input_flux (torch.Tensor or np.ndarray, optional): Input flux to plot at the top. Shape [batch_size, time_steps, input_dim].

    """
    # Convert states to torch.Tensor if it's a numpy array
    if isinstance(states, np.ndarray):
        states = torch.tensor(states)
    elif not isinstance(states, torch.Tensor):
        msg = "states must be a torch.Tensor or np.ndarray"
        raise TypeError(msg)

    if input_flux is not None:
        if isinstance(input_flux, np.ndarray):
            input_flux = torch.tensor(input_flux)
        elif not isinstance(input_flux, torch.Tensor):
            msg = "input_flux must be a torch.Tensor or np.ndarray"
            raise TypeError(msg)

    # Default to all node indices if none are provided
    if node_indices is None:
        node_indices = list(range(states.shape[2]))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    time_steps = states.shape[1]

    if colors is None:
        colors = [None] * len(node_indices)  # Default to None if no colors provided

    for idx, color in zip(node_indices, colors, strict=False):
        avg_state = states[:, :, idx].mean(dim=0)  # Average across batch
        ax.plot(range(time_steps), avg_state, label=f"Node {idx}", color=color)

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("State Value")
    if show_legend:
        ax.legend()
    ax.grid()

    if input_flux is not None:
        ax_flux = ax.inset_axes([0, 1.05, 1, 0.2], transform=ax.transAxes)  # Create an inset axis for the flux plot
        avg_flux = input_flux.mean(dim=0).squeeze()  # Average across batch and remove singleton dimensions
        ax_flux.plot(range(time_steps), avg_flux, color="black")
        ax_flux.set_ylabel("Input")
        ax_flux.set_xticks([])  # Hide x-axis ticks
        ax_flux.grid()

    return fig
