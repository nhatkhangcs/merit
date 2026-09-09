#!/usr/bin/env python3
"""Build or validate the shared greedy one-SQL initial cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from merit.config import load_config
from merit.dataset import load_dataset
from merit.evaluator import Evaluator
from merit.feedback import FeedbackRegime
from merit.generation import (
    InitialCacheIdentity,
    InitialCacheQuery,
    TransformersBackend,
    build_initial_cache,
    load_initial_cache,
    save_initial_cache,
)
from merit.prompts import build_initial_prompt
from merit.runner import installed_package_versions, set_reproducible_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Spider or BIRD JSON config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = load_dataset(config.dataset)
    identity = InitialCacheIdentity.from_config(
        config,
        dataset_checksum=dataset.dataset_checksum,
        database_manifest_hash=dataset.database_manifest_hash,
    )
    prompt_hashes = {
        (example.query_id, example.db_id): build_initial_prompt(
            example.dataset_name,
            example.question,
            example.schema,
            example.evidence,
        ).hash
        for example in dataset
    }
    cache_path = Path(config.initial_cache_path)
    if cache_path.is_file():
        cache = load_initial_cache(
            cache_path,
            expected_identity=identity,
            expected_prompt_hashes=prompt_hashes,
        )
        if set(cache.by_query()) != set(prompt_hashes):
            raise ValueError("existing initial cache does not exactly cover the dataset")
        print(
            f"Validated existing cache with {len(cache.records)} records: "
            f"{cache_path} ({cache.cache_hash})"
        )
        return

    evaluator = Evaluator(
        config.dataset,
        FeedbackRegime.DENOTATION_CONFIRMED.value,
        evaluation_protocol=config.evaluation_protocol,
        reference_timeout_seconds=config.reference_timeout_seconds,
    )
    installed_package_versions(config.package_versions)
    set_reproducible_seeds(config.random_seed)
    examples = {
        (example.query_id, example.db_id): example
        for example in dataset
    }

    def evaluate(query_id: str, db_id: str, sql: str):
        return evaluator.evaluate(examples[(str(query_id), db_id)], sql)

    backend = TransformersBackend(config.model, config.generation)
    cache = build_initial_cache(
        queries=tuple(
            InitialCacheQuery(
                query_id=example.query_id,
                db_id=example.db_id,
                question=example.question,
                schema=example.schema,
                evidence=example.evidence,
            )
            for example in dataset
        ),
        config=config,
        dataset_checksum=dataset.dataset_checksum,
        database_manifest_hash=dataset.database_manifest_hash,
        backend=backend,
        evaluate=evaluate,
    )
    save_initial_cache(cache, cache_path)
    print(
        f"Built {len(cache.records)} greedy initial predictions: "
        f"{cache_path} ({cache.cache_hash})"
    )


if __name__ == "__main__":
    main()
