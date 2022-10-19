#!/usr/env/bin python
"""Experiment testing different values of the truncation parameter T"""

import argparse
import configparser
import os
import sys

sys.path.append("../")

from utils import configure_logger, logger  # noqa
from submit_runs import get_dag  # noqa


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output")
    parser.add_argument("--name", help="Name of the job", default="ins")
    parser.add_argument(
        "--executable",
        "-e",
        default="run_experiment.py",
        help="Exctuable to run",
    )
    parser.add_argument(
        "--n", "-n", type=int, default=5, help="Number of jobs"
    )
    parser.add_argument("--config", type=str, help="Config")
    parser.add_argument("--nlive", type=int, nargs="+")
    return parser.parse_args()


def main():
    """Get and submit the job"""
    configure_logger()
    args = parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    if not args.output:
        output = config["General"]["output"]
    else:
        output = args.output
    base_output = os.path.abspath(output)
    os.makedirs(base_output, exist_ok=True)

    logger.warning(f"Setting up runs for nlive={args.nlive}")
    logger.warning(f"Runs will be saved to {base_output}")
    model_name = config["General"]["model"]
    dims = config["Model"]["dims"]
    for i, nlive in enumerate(args.nlive):
        output = os.path.join(
            base_output, f"{args.name}_{model_name}_{dims}d_nlive{nlive}", ""
        )
        count = 0
        while os.path.exists(output):
            output = os.path.join(
                base_output,
                f"{args.name}_{model_name}_{dims}d_nlive{nlive}_{count}",
                "",
            )
            count += 1
        os.makedirs(output, exist_ok=True)
        config["General"]["output"] = os.path.join(output, "analysis", "")
        config["General"]["plot"] = "False"
        logger.info(f"Setting nlive={nlive}")
        config["Sampler"]["nlive"] = str(nlive)
        updated_config = os.path.join(output, "base_config.ini")
        with open(updated_config, "w") as fp:
            config.write(fp)
        dag = get_dag(output, args, updated_config)
        dag.build_submit()


if __name__ == "__main__":
    main()
