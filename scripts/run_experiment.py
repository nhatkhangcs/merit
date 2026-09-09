#!/usr/bin/env python3
"""Run one MERIT baseline, ablation, or full causal stream."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from merit.config import FEEDBACK_REGIMES, METHODS, load_config
from merit.dataset import load_dataset
from merit.evaluator import Evaluator
from merit.generation import TransformersBackend, load_initial_cache
from merit.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Spider or BIRD JSON config")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--fresh-run",
        action="store_true",
        help="create a new run directory and reject collisions",
    )
    mode.add_argument(
        "--resume",
        metavar="RUN_DIRECTORY",
        help="resume only after every immutable identity field matches",
    )
    parser.add_argument("--run-id", help="optional fresh-run directory identifier")
    parser.add_argument("--method", choices=sorted(METHODS))
    parser.add_argument("--stream-order-seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--feedback-regime", choices=sorted(FEEDBACK_REGIMES))
    args = parser.parse_args()
    if args.resume is not None and args.run_id is not None:
        parser.error("--run-id is valid only with --fresh-run")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        key: value
        for key, value in (
            ("method_name", args.method),
            ("stream_order_seed", args.stream_order_seed),
            ("feedback_regime", args.feedback_regime),
        )
        if value is not None
    }
    config = replace(config, **overrides)
    config.validate()

    dataset = load_dataset(config.dataset)
    cache = load_initial_cache(config.initial_cache_path)
    evaluator = Evaluator(
        config.dataset,
        config.feedback_regime,
        evaluation_protocol=config.evaluation_protocol,
        reference_timeout_seconds=config.reference_timeout_seconds,
    )
    backend = TransformersBackend(config.model, config.generation)
    result = ExperimentRunner(
        config,
        dataset,
        cache,
        evaluator,
        backend,
    ).run(
        fresh_run=args.fresh_run,
        resume_directory=args.resume,
        run_id=args.run_id,
    )
    print(
        f"Completed {config.dataset.name}/{config.method_name} "
        f"stream={config.stream_order_seed}: {result.run_directory}"
    )


if __name__ == "__main__":
    main()
