# Release manifest

This manifest records the scope of the article code release.

## Included

- `src/scp/canary`: deterministic canary generation and source canary-bank handling.
- `src/scp/data`: image-pair and folder-layout data interfaces, including controlled shifts.
- `src/scp/eval`: counterfactual canary and offline evaluation metrics.
- `src/scp/models`: lazy adapters for the model implementations named in the article.
- `src/scp/scoring`: source normalization, score aggregation, and F/I/NF-adjacent decision utilities.
- `src/scp/d9/k2_rules.py`: fixed primary and prospective F/I/NF falsification rules.
- `src/scp/signals`: passive evidence-response signals.
- `src/scp/utils`: typed configuration and reproducibility helpers.
- `src/scp/pipeline.py`: probability-map, passive-score, canary-score, and offline-label stages.
- `scripts/run_article_audit.py` and `scripts/run_pipeline.py`: user-facing execution path.
- `scripts/summarize_audit.py`: source-normalized (c), (e), and F/I/NF summary.
- Focused tests covering canaries, signals, evaluation metrics, scoring, data contracts, controlled shifts, and configuration.

## Intentionally excluded

- Raw datasets and derived data products.
- Model checkpoints and model repositories under `third_party/`.
- TGRS/MDPI manuscript sources, figures, tables, supplementary files, and PDFs.
- Secondary analyses, historical protocol scripts, training utilities, and unrelated project tests.
- Python caches, local environments, logs, and generated results.

The excluded items are either external scientific assets, license-bound dependencies, or development material that is not required by the article audit entry point.
