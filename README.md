# MERIT

**Causal Memory-Augmented Text-to-SQL Repair**

This repository contains the reference implementation and evaluation pipeline for
MERIT, a framework for improving text-to-SQL repair by reusing verified experience
from earlier queries in a causal stream.

This work has been accepted at the
[REALM Workshop at EMNLP](https://realm-workshop.github.io/).

## Overview

MERIT studies whether a text-to-SQL model can improve later predictions by learning
from its earlier repair episodes without accessing future queries or reference SQL.
For each query, the system:

1. loads a shared deterministic initial SQL prediction;
2. executes the prediction and observes the permitted feedback;
3. classifies the current failure when repair is needed;
4. retrieves causally available positive and negative repair memories;
5. generates and evaluates a revised SQL query; and
6. adds finalized evidence to memory only after the episode is complete.

The default `merit_full` policy uses error-type filtering, separate positive and
negative memory pools, and hybrid retrieval combining dense similarity (0.75) with
BM25 similarity (0.25). It retrieves at most three positive and one negative memory
per repair step. The supplied configurations allow up to seven repair generations.

### Reproducibility properties

- **Shared initialization:** all compared methods begin with exactly the same greedy
  initial SQL prediction for each example.
- **Causal memory:** a query at stream position `t` may retrieve only memories created
  by completed episodes at positions `< t`; the current query is excluded.
- **Reference isolation:** reference SQL and reference denotations are confined to the
  evaluation layer and are never placed in retrieval memory or generation prompts.
- **Immutable identities:** model, tokenizer, embedding model, evaluation protocol,
  datasets, databases, configuration, and source artifacts are pinned or hashed.
- **Auditable runs:** predictions, trajectories, retrieval decisions, memory entries,
  resource accounting, and official scores are persisted as separate artifacts.
- **Safe execution:** generated SQL is evaluated against SQLite databases opened in
  read-only mode with write operations denied and execution deadlines enforced.

## Supported experiments

The same runner supports Spider and BIRD, along with the following methods.

| Group | Methods | Description |
| --- | --- | --- |
| Main method | `merit`, `merit_full` | Causal typed hybrid retrieval over positive and negative repair evidence. |
| Baselines | `zeroshot`, `iterative`, `vanilla`, `reflexion`, `dynamic_rag` | No-repair, local repair, fixed-example, reflection, and untyped dynamic-retrieval controls. |
| Ablations | `positive_only`, `no_type_filter`, `confidence_aware_type_filter`, `no_dense_rerank`, `no_bm25`, `random_same_type`, `cross_database_only` | Controlled changes to memory polarity, type filtering, ranking, and database scope. |
| Non-causal control | `transductive_batch` | Synchronous batch-memory control; this method must not be reported as online learning. |

Two feedback regimes are implemented:

- `denotation_confirmed`: correctness is confirmed against the evaluation target.
- `dbms_only`: only DBMS execution feedback is exposed; oracle correctness is not
  available to the online method.

## Repository structure

| Path | Purpose |
| --- | --- |
| `merit/` | Core configuration, generation, evaluation, classification, memory, retrieval, runner, metrics, and export modules. |
| `configs/` | Reproducible Spider/BIRD configurations, stream-order seeds, and fixed examples for the vanilla baseline. |
| `scripts/build_initial_cache.py` | Builds or validates the shared greedy initial-prediction cache. |
| `scripts/run_experiment.py` | Runs or resumes a baseline, ablation, or MERIT stream. |
| `scripts/export_results.py` | Validates completed runs and exports comparison tables. |
| `scripts/reannotate_initial_cache.py` | Migrates and reannotates a supported legacy cache under the current official protocol. |
| `official_evaluation/` | Post-hoc Spider and BIRD official-evaluation wrappers and protocol documentation. |

Datasets, model weights, initial caches, run directories, and generated results are
not distributed in this repository.

## Environment setup

Python 3.11 and a CUDA-capable Linux environment are recommended for reproducing the
provided GPU configurations. They use Qwen2.5-7B-Instruct with 4-bit NF4
quantization, BGE-large-en-v1.5 embeddings, CUDA 12.8 PyTorch, and GPU FAISS.

```bash
git clone https://github.com/nhatkhangcs/merit.git
cd merit

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt
```

The experiment runner checks the exact package versions declared in each JSON
configuration. A reportable run will stop if a required package is missing or its
version differs from the pinned environment. Model and embedding weights are loaded
from Hugging Face at immutable revisions; ensure they are accessible online or
already present in the local Hugging Face cache.

## Dataset preparation

Obtain Spider and BIRD from their official distributions and follow their respective
licenses and terms of use. Place the extracted files at the paths below, or update a
copy of the corresponding configuration with explicit local paths.

| Dataset | Required paths in the supplied configuration |
| --- | --- |
| Spider | `data/spider/dev.json`, `data/spider/tables.json`, `data/spider_test_suite/database/` |
| BIRD | `data/BIRD/dev.json`, `data/BIRD/dev_tables.json`, `data/BIRD/dev_databases/`, and `data/BIRD/dev.sql` for post-hoc official scoring |

- Spider: [dataset and benchmark repository](https://github.com/taoyds/spider)
- BIRD: [benchmark website](https://bird-bench.github.io/)

The loader computes dataset and database-manifest hashes. Existing caches and resumed
runs are rejected if those identities do not match the current data.

## Official evaluator assets

Post-hoc scoring is bound to the following upstream evaluator revisions:

- Spider test-suite evaluator:
  [`e97acc546ecbee8fa27fa8dbf025ef61493a876c`](https://github.com/taoyds/test-suite-sql-eval/commit/e97acc546ecbee8fa27fa8dbf025ef61493a876c)
- BIRD evaluator:
  [`483554eae102996f5ec1f4feab4e78ef29c2a394`](https://github.com/AlibabaResearch/DAMO-ConvAI/commit/483554eae102996f5ec1f4feab4e78ef29c2a394)

The official-evaluation wrappers expect the corresponding source and dependency
bundle under `official_evaluation/vendor/`. These third-party assets are not fetched
automatically. Their source hashes, dependency versions, timeouts, and evaluator
options are validated before scoring. See
[`official_evaluation/README.md`](official_evaluation/README.md) for the evaluation
contract and commands.

## Reproducing experiments

Run all commands from the repository root.

### 1. Build or validate the shared initial cache

```bash
python scripts/build_initial_cache.py --config configs/spider.json
python scripts/build_initial_cache.py --config configs/bird.json
```

If the configured cache does not exist, the command generates one deterministic
initial SQL prediction per query. If it already exists, the command validates its
identity, hashes, prompt coverage, and annotations instead of regenerating it.

Before launching the full experiment matrix, validate the cache labels with the
pinned official evaluator:

```bash
python official_evaluation/evaluate_cache.py \
  --config configs/spider.json \
  --cache cache/spider_qwen2.5-7b_official-v1.json

python official_evaluation/evaluate_cache.py \
  --config configs/bird.json \
  --cache cache/bird_qwen2.5-7b_official-v1.json
```

### 2. Run MERIT

The paper protocol uses stream-order seeds 0, 1, and 2. For example:

```bash
for seed in 0 1 2; do
  python scripts/run_experiment.py \
    --config configs/spider.json \
    --fresh-run \
    --method merit_full \
    --stream-order-seed "$seed"
done
```

Replace `configs/spider.json` with `configs/bird.json` for BIRD, or select another
method from the table above. A timestamped run directory is created beneath the
configured `run_root` and printed on completion.

Use a unique explicit identifier when integrating with a scheduler:

```bash
python scripts/run_experiment.py \
  --config configs/spider.json \
  --fresh-run \
  --method merit_full \
  --stream-order-seed 0 \
  --run-id spider-merit-full-s0
```

### 3. Resume an interrupted run

```bash
python scripts/run_experiment.py \
  --config configs/spider.json \
  --resume runs/spider/spider-merit-full-s0 \
  --method merit_full \
  --stream-order-seed 0
```

Resume is deliberately strict: all immutable configuration, source, data, model,
cache, protocol, and package identities must match the original run.

### 4. Run post-hoc official evaluation

```bash
python official_evaluation/evaluate_run.py \
  runs/spider/spider-merit-full-s0
```

The evaluator hashes immutable run artifacts before and after scoring, writes only
`official_eval.json`, and reports a mismatch if the official correct count differs
from the internally recorded count. BIRD uses its pinned one-worker, 30-second
protocol; Spider uses the pinned test-suite execution evaluator.

### 5. Export comparison tables

Pass the completed run directories for the methods and stream orders being compared:

```bash
python scripts/export_results.py \
  runs/spider/<run-directory-1> \
  runs/spider/<run-directory-2> \
  runs/spider/<run-directory-3> \
  --output-dir results/spider
```

Reportable exports require all three stream-order seeds by default. The command
validates the runs and writes `results.csv` and `results.md`.

## Run artifacts

A completed run directory contains the following auditable outputs:

| Artifact | Contents |
| --- | --- |
| `config.json` | Fully resolved experiment configuration. |
| `manifest.json` | Source, data, model, package, protocol, cache, and run identities. |
| `predictions.jsonl` | One final SQL prediction per query. |
| `trajectories.jsonl` | Initial and repaired SQL, outcomes, classifications, and accounting per episode. |
| `retrieval_log.jsonl` | Candidate-pool sizes, retrieved memory IDs, provenance, and retrieval scores. |
| `memory_positive.jsonl` / `memory_negative.jsonl` | Persisted verified repair evidence and observed failed directions for memory-enabled methods. |
| `metrics.json` | Success@1, final accuracy, repair counts, failure breakdowns, diagnostics, and resource totals. |
| `official_eval.json` | Status and results of pinned post-hoc official evaluation. |
| `stdout.log` | Human-readable episode progress. |

## Reporting guidance

- Compare methods only when they share the same initial-cache identity.
- Aggregate reportable results over stream-order seeds 0, 1, and 2.
- Report the feedback regime and evaluator protocol with every result.
- Treat `transductive_batch` only as a non-causal batch control, not as online
  learning.
- Preserve run directories: their manifests and per-query traces are part of the
  reproducibility record.

## Acknowledgements

This work was accepted at the
[REALM Workshop at EMNLP](https://realm-workshop.github.io/). We acknowledge the
authors and maintainers of Spider, BIRD, Qwen, BGE, and the open-source libraries and
official evaluators used by this project. The final bibliographic record and BibTeX
entry will be added when available.

## Citation

BibTeX coming soon.

## License and third-party assets

Repository license information will be added. Dataset files, model weights, and
official evaluator assets remain subject to their original licenses and terms of use.
