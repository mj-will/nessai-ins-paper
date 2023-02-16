#!/usr/env/bin python
"""Python script for submitting runs via HTCondor"""
import argparse
import configparser
import os
import sys

from pycondor import Job, Dagman

sys.path.append("../")

from utils import configure_logger, logger  # noqa: E402


EXTRA_LINES = [
    "checkpoint_exit_code=130",
    "max_retries=5",
    "accounting_group=ligo.dev.o4.cbc.pe.bilby",
]


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output")
    parser.add_argument("--name", help="Name of the job", default="ins")
    parser.add_argument("--executable", "-e", help="Exctuable to run")
    parser.add_argument(
        "--n", "-n", type=int, default=1, help="Number of jobs"
    )
    parser.add_argument("--config", type=str, help="Config")
    parser.add_argument("--dims", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def get_dag(output, args, config_path):
    """Get the DAG"""
    name = f"{args.name}"
    submit = os.path.join(output, "submit", "")
    dag = Dagman(name, submit=submit)
    log_path = os.path.join(output, "analysis_logs", "")
    os.makedirs(log_path, exist_ok=True)
    for i in range(args.n):
        tag = f"run_{i}"
        job_name = f"{name}_{tag}"
        extra_lines = EXTRA_LINES + [
            f"log={log_path}{tag}.log",
            f"output={log_path}{tag}.out",
            f"error={log_path}{tag}.err",
        ]
        job = Job(
            name=job_name,
            executable=args.executable,
            queue=1,
            getenv=True,
            submit=submit,
            request_memory="2GB",
            request_cpus=1,
            extra_lines=extra_lines,
        )
        job.add_arg(
            f"--config={config_path} --tag={tag} --seed={args.seed + i}"
        )
        dag.add_job(job)
    return dag


def main():
    """Get and submit the job"""
    configure_logger()
    args = parse_args()
    base_output = os.path.abspath(args.output)
    os.makedirs(base_output, exist_ok=True)

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(args.config)
    model_name = config["General"]["model"]
    logger.warning(f"Setting up runs for dims={args.dims}")
    for d in args.dims:
        output = os.path.join(
            base_output, f"{args.name}_{model_name}_{d}d", ""
        )
        count = 0
        while os.path.exists(output):
            output = os.path.join(
                base_output, f"{args.name}_{model_name}_{d}d_{count}", ""
            )
            count += 1
        os.makedirs(output, exist_ok=True)
        config["General"]["output"] = os.path.join(output, "analysis", "")
        config["General"]["plot"] = "False"
        config["Model"]["dims"] = str(d)
        updated_config = os.path.join(output, "base_config.ini")
        with open(updated_config, "w") as fp:
            config.write(fp)
        dag = get_dag(output, args, updated_config)
        dag.build_submit()


if __name__ == "__main__":
    main()
