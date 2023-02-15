#!/usr/bin/env python
"""Plot the meta proposal.

Michael J. Williams 2023
"""
import os
import sys

import bilby
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

basedir = "../../"

sys.path.append(basedir)

from utils import configure_plotting

configure_plotting(basedir)

# Disable the bilby plotting style
os.environ["BILBY_STYLE"] = "none"


def main(result_file):

    result = bilby.core.result.read_in_result(result_file)

    os.makedirs("figures", exist_ok=True)

    samples = result.nested_samples
    its = np.unique(samples["iteration"])
    colours = plt.get_cmap("viridis")(np.linspace(0, 1, len(its)))
    vmax = np.max(samples["log_likelihood"])
    vmin = min(
        vmax - 0.10 * np.ptp(samples["log_likelihood"]),
        samples["log_likelihood"][samples["iteration"] == its[-1]].min(),
    )

    include = np.arange(4, len(its), 1)

    fig, axs = plt.subplots(1, 2, sharey=False)

    for it, c in zip(its, colours):
        current = samples["iteration"] == it
        if it in include:
            sns.kdeplot(
                x=samples["chirp_mass"][current],
                y=samples["mass_ratio"][current],
                levels=[0.1],
                ax=axs[0],
                color=c,
            )

        data = samples["log_likelihood"][current]
        axs[1].hist(
            data,
            "auto",
            histtype="step",
            color=c,
            density=True,
            cumulative=False,
        )

    axs[0].axvline(1.4492961609281714, c="tab:red")
    axs[0].axhline(0.9316375554581786, c="tab:red")

    # axs[0].set_xlim([1.342689302683192, 1.542689302683192])
    axs[0].set_xlim([1.445, 1.453])
    axs[0].set_ylim([0.5, 1.0])
    axs[0].set_xlabel(r"$\mathcal{M}\,[M_{\odot}]$")
    axs[0].set_ylabel(r"$q$")

    axs[1].set_xlabel(r"$\ln \mathcal{L}$")
    axs[1].set_ylabel(r"$p(\ln \mathcal{L})$")
    # axs[1].set_xlim(vmin, vmax)
    axs[1].set_yscale("log")
    axs[1].set_ylim(5e-4, 3)
    plt.tight_layout()
    fig.savefig("figures/bns_likelihood_levels.pdf")


if __name__ == "__main__":
    main(sys.argv[1])
