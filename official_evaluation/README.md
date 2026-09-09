# Post-hoc official evaluation

`evaluate_run.py` evaluates completed MERIT predictions without changing the
run configuration, manifest, predictions, trajectories, or metrics. It hashes
those artifacts before and after evaluation, writes only
`official_eval.json`, and exits with status 2 after writing when the official
correct count differs from the internal count.

The vendored evaluators are pinned to:

- Spider test-suite evaluator commit
  `e97acc546ecbee8fa27fa8dbf025ef61493a876c`
- BIRD evaluator commit
  `483554eae102996f5ec1f4feab4e78ef29c2a394`

Before starting GPU runs, independently validate each migrated v5 cache:

```bash
python official_evaluation/evaluate_cache.py \
  --config configs/spider.json \
  --cache cache/spider_qwen2.5-7b_official-v1.json

python official_evaluation/evaluate_cache.py \
  --config configs/bird.json \
  --cache cache/bird_qwen2.5-7b_official-v1.json
```

The cache validator writes an exclusive `.official_validation.json` sidecar
and exits successfully only when every per-query correctness label matches. It
uses one BIRD worker, as pinned by the protocol, and does not load a model or
GPU.

After a run completes, score its final predictions from the repository root:

```bash
python official_evaluation/evaluate_run.py \
  runs/spider/spider-merit-full-s0

python official_evaluation/evaluate_run.py \
  runs/bird/bird-merit-full-s0
```

Spider defaults to `data/spider_test_suite/database` and retains the official
archive hash from `data/_downloads/spider_test_suite.zip`. BIRD defaults to
the configured database root and the `dev.sql` beside its configured
`dev.json`. Use `--help` to inspect explicit overrides. Reportable BIRD scoring uses
the protocol-bound one-worker, 30-second setting.

Predictions are validated in their deterministic shuffled stream order, then
reordered by authoritative `source_index` for native evaluator inputs. BIRD
SQL remains intact inside its native JSON format. Spider comments and
whitespace tokens are converted to its required one-query-per-line format
without modifying string literals.
