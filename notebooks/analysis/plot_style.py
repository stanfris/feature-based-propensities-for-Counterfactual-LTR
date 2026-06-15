THESIS_FONT_SIZES = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}

RESULT_COMPARISON_FONT_SIZES = {
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}

DENSE_TICK_LABEL_SIZE = 8
DENSE_ANNOTATION_SIZE = 9
DENSE_PANEL_TITLE_SIZE = 12
DENSE_SUPTITLE_SIZE = 15


def apply_thesis_plot_style(plt):
    plt.rcParams.update(THESIS_FONT_SIZES)


def apply_result_comparison_plot_style(plt):
    plt.rcParams.update(RESULT_COMPARISON_FONT_SIZES)
