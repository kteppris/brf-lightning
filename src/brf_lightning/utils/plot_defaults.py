import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import math

# IEEE-styled figure generator
def new_figure(width_pt=426.8, fraction=1.0, subplots=(1, 1), base_fontsize=10, height_per_subplot=None):

    inches_per_pt = 1 / 72.27
    width_in = width_pt * fraction * inches_per_pt
    golden = (math.sqrt(5) - 1) / 2
    if height_per_subplot is None:
        height_in = width_in * golden * (subplots[0] / subplots[1])
    else:
        height_in = subplots[0] * height_per_subplot

    scaled_fontsize = base_fontsize * fraction
    mpl.rcParams.update({'font.size': scaled_fontsize})

    fig, axs = plt.subplots(*subplots, figsize=(width_in, height_in), squeeze=False)

    if subplots == (1, 1):
        return fig, axs[0, 0]
    else:
        return fig, axs.flatten()



def init_plot_style(base_fontsize=10):
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Use your preferred display style
    plt.style.use(['science', 'ieee'])

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize * 0.8,
        "ytick.labelsize": base_fontsize * 0.8,
        "legend.fontsize": base_fontsize * 0.8,
        "text.usetex": False,  # For display only
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    })

    # Restore color cycle (optional)
    default_colors = mpl.rcParamsDefault['axes.prop_cycle'].by_key()['color']
    mpl.rcParams['axes.prop_cycle'] = cycler(color=default_colors)

def export_figure_to_pgf(fig, filename):
    import matplotlib.pyplot as plt

    with plt.rc_context({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.titlesize": 11,
        "lines.linewidth": .6,
    }):
        fig.savefig(
            filename,
            backend='pgf',
            bbox_inches="tight"        
        )
        print(f"[PGF Export] Saved figure to '{filename}'.")

