import pandas as pd
from itertools import product
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import binom, kstest, combine_pvalues

cbc_param_labels = {
    "a_1": "$\\chi_1$",
    "a_2": "$\\chi_2$",
    "chirp_mass": r"$\mathcal{M}$",
    "dec": "$\\delta$",
    "ra": "$\\alpha$",
    "geocent_time": "$t_\textrm{c}$",
    "luminosity_distance": "$d_\textrm{L}$",
    "mass_ratio": "$q$",
    "tilt_1": "$\\theta_1$",
    "tilt_2": "$\\theta_2$",
    "phi_12": "$\\phi_{12}$",
    "phi_jl": "$\\phi_{JL}$",
    "psi": "$\\psi$",
    "theta_jn": "$\\theta_{JN}$",
}


def make_pp_plot(
    results,
    confidence_interval=[0.68, 0.95, 0.997],
    lines=None,
    legend_fontsize="x-small",
    keys=None,
    title=True,
    confidence_interval_alpha=0.1,
    labels=None,
    height=None,
    width=None,
    include_legend=True,
    weight_list=None,
    palette="RdYlBu",
    **kwargs
):
    """
    Make a P-P plot for a set of runs with injected signals.

    Parameters
    ----------
    results: list
        A list of Result objects, each of these should have injected_parameters
    filename: str, optional
        The name of the file to save, the default is "outdir/pp.png"
    save: bool, optional
        Whether to save the file, default=True
    confidence_interval: (float, list), optional
        The confidence interval to be plotted, defaulting to 1-2-3 sigma
    lines: list
        If given, a list of matplotlib line formats to use, must be greater
        than the number of parameters.
    legend_fontsize: float
        The font size for the legend
    keys: list
        A list of keys to use, if None defaults to search_parameter_keys
    confidence_interval_alpha: float, list, optional
        The transparency for the background condifence interval
    kwargs:
        Additional kwargs to pass to matplotlib.pyplot.plot

    Returns
    -------
    fig, pvals:
        matplotlib figure and a NamedTuple with attributes `combined_pvalue`,
        `pvalues`, and `names`.
    """

    if keys is None:
        keys = results[0].search_parameter_keys

    if weight_list is None:
        weight_list = [None] * len(results)

    credible_levels = list()
    for i, result in enumerate(results):
        credible_levels.append(
            result.get_all_injection_credible_levels(
                keys, weights=weight_list[i]
            )
        )
    credible_levels = pd.DataFrame(credible_levels)

    if lines is None:
        colors = sns.color_palette(palette, n_colors=6)
        linestyles = ["-", "--", ":"]
        style = list(product(linestyles, colors))

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    default_height, default_width = plt.rcParams["figure.figsize"]
    if width is None:
        width = default_width
    if height is None:
        height = default_height
    fig, ax = plt.subplots(figsize=(height, width))

    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(
            confidence_interval
        )
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval"
        )

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1.0 - ci) / 2.0
        lower = binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color="grey")

    pvalues = []
    print("Key: KS-test p-value")
    for ii, key in enumerate(credible_levels):
        pp = np.array(
            [
                sum(credible_levels[key].values < xx) / len(credible_levels)
                for xx in x_values
            ]
        )
        pvalue = kstest(credible_levels[key], "uniform").pvalue
        pvalues.append(pvalue)
        print("{}: {}".format(key, pvalue))

        try:
            name = labels[key]
        except AttributeError:
            name = key
        label = "{} ({:2.3f})".format(name, pvalue)
        plt.plot(
            x_values,
            pp,
            ls=style[ii][0],
            c=style[ii][1],
            label=label,
            **kwargs
        )

    Pvals = namedtuple("pvals", ["combined_pvalue", "pvalues", "names"])
    pvals = Pvals(
        combined_pvalue=combine_pvalues(pvalues)[1],
        pvalues=pvalues,
        names=list(credible_levels.keys()),
    )
    print("Combined p-value: {}".format(pvals.combined_pvalue))

    if title:
        ax.set_title(
            "N={}, p-value={:2.4f}".format(
                len(credible_levels), pvals.combined_pvalue
            )
        )
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    if include_legend:
        ax.legend(
            handlelength=2,
            labelspacing=0.25,
            frameon=False,
            fontsize=legend_fontsize,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig, pvals
