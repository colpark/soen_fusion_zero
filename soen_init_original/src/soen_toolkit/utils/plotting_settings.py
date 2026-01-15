# FILEPATH: soen/utils/plotting_settings.py


import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns


def apply_plot_settings(use_tex: bool = False, grid: bool = True, disable_warnings: bool = True) -> None:
    """Apply consistent plot settings for all visualizations.

    Args:
        use_tex (bool): Use LaTeX for rendering text (if available).
        grid (bool): Enable grid lines on plots.
        disable_warnings (bool): Disable font-related warnings.

    """
    if disable_warnings:
        warnings.filterwarnings("ignore", message=".*findfont.*")

    # Check if Times New Roman is available
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    serif_fonts = ["Times New Roman", "DejaVu Serif", "Liberation Serif", "Nimbus Roman", "Serif"]

    # Find the first available serif font
    font_to_use = next((font for font in serif_fonts if font in available_fonts), "serif")

    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": [font_to_use],
            "axes.labelsize": 14,
            "font.size": 14,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.dpi": 300,
            "figure.figsize": [12, 5],
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.grid": grid,
            "grid.alpha": 0.7,
            "grid.linestyle": "--",
        }
    )

    sns.set_palette("dark")


def save_and_show_plot(filename=None, fig=None, file_format="svg", save=True, show=True) -> None:
    """Save and/or show the plot with consistent settings.

    Args:
        filename (str, optional): The filename to save the plot. Required if save is True.
        fig (matplotlib.figure.Figure, optional): The figure to save. Defaults to the current figure.
        file_format (str): The file format to save the plot. Defaults to 'svg'.
        save (bool): Whether to save the plot. Defaults to True.
        show (bool): Whether to show the plot. Defaults to True.

    """
    if fig is None:
        fig = plt.gcf()

    if save:
        if filename is None:
            msg = "Filename must be provided if save is True."
            raise ValueError(msg)
        full_filename = f"{filename}.{file_format}"
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(full_filename, format=file_format, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
