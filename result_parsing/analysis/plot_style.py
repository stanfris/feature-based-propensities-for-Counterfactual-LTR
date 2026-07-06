PLOT_FONT_SIZES = {
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Linux Libertine", "Libertinus Serif"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}

RESULT_COMPARISON_FONT_SIZES = {
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Linux Libertine", "Libertinus Serif"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
    "font.size": 17,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}

DENSE_TICK_LABEL_SIZE = 8
DENSE_ANNOTATION_SIZE = 9
DENSE_PANEL_TITLE_SIZE = 12
DENSE_SUPTITLE_SIZE = 15


def apply_plot_style(plt):
    plt.rcParams.update(PLOT_FONT_SIZES)


def apply_result_comparison_plot_style(plt):
    plt.rcParams.update(RESULT_COMPARISON_FONT_SIZES)
