"""
General utilities for nessai importance nested sampling paper.
"""
import argparse
from ast import literal_eval
import configparser
import glob
import json
import logging
import os
import re
import shutil
from typing import Any, Callable, Optional, Tuple

import nessai
from nessai.flowsampler import FlowSampler
from nessai.flowmodel import update_config
from nessai.utils import setup_logger, NessaiJSONEncoder
from nessai_models import (
    Gaussian,
    Rosenbrock,
    GaussianMixture,
    GaussianMixtureWithData,
    Pyramid,
    EggBox,
)
import numpy as np
import pandas as pd
import torch
import tqdm


logger = logging.getLogger("ins-experiment")


def parse_args() -> argparse.ArgumentParser:
    """Parse command line args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for outdir",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=150914)
    parser.add_argument("--summary", type=bool, default=True)
    return parser.parse_args()


def configure_logger(level="INFO"):
    """Configure the logger."""
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)-8s: %(message)s",
            datefmt="%m-%d %H:%M",
        )
    )
    stream_handler.setLevel("INFO")
    logger.addHandler(stream_handler)


def natural_sort(values):
    """Natural sort a list.

    Based on: https://stackoverflow.com/a/4836734
    """
    # fmt: off
    convert = (lambda text: int(text) if text.isdigit() else text.lower())  # noqa: 731
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa: 731
    # fmt: on
    return sorted(values, key=alphanum_key)


def find_results_files(path, file="result.json"):
    """Find all of the results files"""
    p = path + "/**/" + file
    logger.info(f"Searching for: {p}")
    files = glob.glob(p, recursive=True)
    logger.info(f"Found {len(files)} results files")
    return files


def load_results(results_files):
    """Load the log-evidence"""
    data = []
    for i, rf in tqdm.tqdm(enumerate(results_files)):
        try:
            with open(rf, "r") as f:
                d = json.load(f)
            data.append(d)
        except json.JSONDecodeError:
            print(f"Skipping: {rf}")
    df = pd.DataFrame(data)
    return df


def load_json(filename: str) -> dict:
    """Load a JSON file"""
    with open(filename, "r") as fp:
        d = json.load(fp)
    return d


class HyperCubeMixin:
    """Mixin that adds hyper-cube methods to a nessai Model"""

    def to_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (x[n] - self.bounds[n][0]) / (
                self.bounds[n][1] - self.bounds[n][0]
            )
        return x_out

    def from_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (self.bounds[n][1] - self.bounds[n][0]) * x[
                n
            ] + self.bounds[n][0]
        return x_out


def from_string(value: str) -> Any:
    """Convert the value from a string.

    Uses `ast.literal` eval to convert a value from a string. If a ValueError
    is raised, the string is returned.
    """
    try:
        return literal_eval(value)
    except ValueError:
        return value


def get_config(config_file: str) -> configparser.ConfigParser:
    """Get the config from a config file."""
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file)
    if config.getboolean("General", "use_float64", fallback=False):
        torch.set_default_dtype(torch.float64)
    if eps := config.getfloat("General", "eps", fallback=None):
        nessai.config.eps = eps
    return config


def get_model_class(name: str, dims: int) -> Tuple[Callable, dict]:
    """Get the class for the from the name and any kwargs."""
    kwargs = {}
    if name == "rosenbrock":
        Model = Rosenbrock
    elif name == "gaussian":
        Model = Gaussian
    elif name == "gmm":
        Model = GaussianMixture
    elif name == "gmm_paper":
        Model = GaussianMixture
        kwargs["config"] = [
            {
                "mean": np.concatenate([[0, 4], np.zeros(dims - 2)]),
                "cov": np.eye(dims),
            },
            {
                "mean": np.concatenate([[0, -4], np.zeros(dims - 2)]),
                "cov": np.eye(dims),
            },
            {
                "mean": np.concatenate([[4, 0], np.zeros(dims - 2)]),
                "cov": np.eye(dims),
            },
            {
                "mean": np.concatenate([[-4, 0], np.zeros(dims - 2)]),
                "cov": np.eye(dims),
            },
        ]
        kwargs["weights"] = [0.4, 0.3, 0.2, 0.1]
        kwargs["n_gaussians"] = 4
    elif name == "gmm_hard":
        Model = GaussianMixture
        kwargs["config"] = [
            {
                "mean": np.concatenate([[0, 4], np.zeros(dims - 2)]),
                "cov": 0.2 * np.eye(dims),
            },
            {
                "mean": np.concatenate([[0, -4], np.zeros(dims - 2)]),
                "cov": 0.1 * np.eye(dims),
            },
            {
                "mean": np.concatenate([[4, 0], np.zeros(dims - 2)]),
                "cov": 0.05 * np.eye(dims),
            },
            {
                "mean": np.concatenate([[-4, 0], np.zeros(dims - 2)]),
                "cov": 0.1 * np.eye(dims),
            },
        ]
        kwargs["weights"] = [0.4, 0.3, 0.2, 0.1]
        kwargs["n_gaussians"] = 4
    elif name == "gmm_data":
        Model = GaussianMixtureWithData
    elif name == "eggbox":
        Model = EggBox
    elif name == "pyramid":
        Model = Pyramid
    else:
        raise ValueError(f"Unknown model: {name}")

    return Model, kwargs


def get_model(
    config: configparser.ConfigParser, **kwargs
) -> nessai.model.Model:
    """Get the model from the config"""
    dims = config.getint("Model", "dims")
    model_kwargs = {}
    for k, v in config["Model"].items():
        if k == "dims":
            continue
        else:
            model_kwargs[k] = from_string(v)
    kwargs.update(model_kwargs)
    ModelClass, additional_kwargs = get_model_class(
        config["General"]["model"].lower(), dims
    )
    kwargs.update(additional_kwargs)
    print(kwargs)
    logger.info(f"Model config: dims={dims}, kwargs={kwargs}")
    model = ModelClass(dims, **kwargs)
    return model


def get_flow_config(config: configparser.ConfigParser) -> dict:
    """Get the flow config dictionary from the config.

    Uses the default config from nessai and then updates values given the
    config.
    """
    base_config = update_config(
        dict(model_config=dict(n_inputs=config.getint("Model", "dims")))
    )

    for setting, value in config["Flow"].items():
        if setting == "kwargs":
            raise ValueError
        if setting in base_config:
            base_config[setting] = from_string(value)
        elif setting in base_config["model_config"]:
            base_config["model_config"][setting] = from_string(value)
        else:
            base_config["model_config"]["kwargs"][setting] = from_string(value)

    logger.info(f"Flow config: {base_config}")
    if base_config["model_config"]["ftype"] == "maf":
        base_config["model_config"]["kwargs"].pop("linear_transform", None)
        base_config["model_config"]["kwargs"].pop("pre_transform", None)

    return base_config


def get_sampler(
    model: nessai.model.Model, output: str, config: configparser.ConfigParser
) -> FlowSampler:
    """Get the configure sampler."""
    kwargs = config["Sampler"]
    print(list(kwargs.keys()))
    fixed_kwargs = {}
    for k, v in kwargs.items():
        fixed_kwargs[k] = from_string(v)

    importance_nested_sampler = config.getboolean(
        "General", "importance_nested_sampler"
    )
    logger.info(f"Importance sampler: {importance_nested_sampler}")
    logger.info(f"Kwargs for sampler: \n {fixed_kwargs}")
    flow_config = get_flow_config(config)
    logger.info(f"Flow config: \n {flow_config}")
    logger.info("Getting sampler")

    ins = FlowSampler(
        model,
        resume=config.getboolean("General", "resume", fallback=False),
        output=output,
        importance_nested_sampler=importance_nested_sampler,
        flow_config=flow_config,
        seed=config.getint(
            "General", "seed", fallback=np.random.randint(0, 1e4)
        ),
        plot=config.getboolean("General", "plot", fallback=False),
        **fixed_kwargs,
    )
    return ins


def copy_config_file(
    config_file: str, output: str, filename: str = "config.ini"
) -> None:
    """Copy the config file to output directory"""
    os.makedirs(output, exist_ok=True)
    dest = os.path.join(output, filename)
    shutil.copyfile(config_file, dest)


def get_output(config, tag=None):
    base = config.get("General", "output")
    if tag is not None:
        output = f"{base}{tag}"
    else:
        name = (
            f"{config.get('General', 'model').lower()}"
            f"_{config.get('Model', 'dims')}d"
        )
        count = 0
        while True:
            output = os.path.join(base, f"{name}_dev{count}", "")
            result = os.path.join(output, "result.json")
            if not os.path.exists(output) or not os.path.exists(result):
                break
            count += 1
        if not config.getboolean("General", "increment_dir") and count > 0:
            count -= 1
            output = os.path.join(base, f"{name}_dev{count}", "")
    return output


def get_post_processing_kwargs(config: configparser.ConfigParser) -> dict:
    """Get the post processing keyword arguments"""
    kwargs = {}
    for k, v in config["Postprocessing"].items():
        kwargs[k] = from_string(v)
    return kwargs


def save_summary(
    sampler: FlowSampler, filename: str, extra: Optional[dict] = None
) -> None:
    """Save a summary of run with only the main information"""
    summary = {}
    if extra:
        logger.info(f"Including: {extra} in summary")
        summary.update(extra)

    summary["log_evidence"] = sampler.ns.log_evidence
    summary["log_evidence_error"] = sampler.ns.log_evidence_error
    summary["sampling_time"] = sampler.ns.sampling_time.total_seconds()
    summary["likelihood_evaluations"] = sampler.ns.model.likelihood_evaluations
    summary["ess"] = sampler.ns.posterior_effective_sample_size

    with open(filename, "w") as fp:
        json.dump(summary, fp, cls=NessaiJSONEncoder, indent=4)


def run_sampler(
    config_file, log_level="INFO", seed=None, summary=True, tag=None
):
    """Run the sampler"""
    config = get_config(config_file)
    output = get_output(config, tag=tag)
    copy_config_file(config_file, output)

    plot = config.getboolean("General", "plot", fallback=False)
    logger.warning(f"Output file: {output}")
    logger.warning(f"Plot={plot}")

    if seed:
        config["General"]["seed"] = str(seed)

    setup_logger(output=output, log_level=log_level)
    model = get_model(config)
    sampler = get_sampler(model, output, config)
    sampler.run(plot=plot, save=save)

    if hasattr(model, "truth"):
        logger.info(f"True log-evidence: {model.truth}")

    if summary:
        d = dict()
        if config.getboolean("General", "importance_nested_sampler") is True:
            d["final_log_evidence"] = sampler.ns.final_state.log_evidence
            d[
                "final_log_evidence_error"
            ] = sampler.ns.final_state.log_evidence_error
        save_summary(sampler, os.path.join(output, "summary.json"), extra=d)


def run_basic_experiment():
    """Run an experiment from start to finish"""
    configure_logger()
    args = parse_args()
    logger.info(f"Setting random seed to: {args.seed}")
    if args.seed:
        seed = np.random.seed(args.seed)
    else:
        seed = None
    if hasattr(args, "tag"):
        tag = args.tag
    else:
        tag = None
    run_sampler(
        args.config,
        log_level=args.log_level,
        seed=seed,
        summary=args.summary,
        tag=tag,
    )
