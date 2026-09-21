# Canary Change Probe

Code release for *Label-Free Falsification of Silent Failures in Remote-Sensing Change Detection*.

## What is included

This repository contains only the code needed to reproduce the article's canary and passive-response audit. It includes the data loaders, model-adapter interface, canary generators, signal computations, evaluation metrics, scoring utilities, configuration loader, pipeline, focused tests, and the primary command-line entry point.

The repository does not contain remote-sensing datasets, model checkpoints, third-party model repositories, manuscript sources, PDFs, or generated results. Those items remain subject to their original licenses and must be obtained separately.

## Requirements

- Python 3.10 or newer.
- NumPy, SciPy, scikit-learn, Pillow, PyYAML, and pytest.
- PyTorch and the matching upstream model repositories for a full inference run.
- The datasets and checkpoints listed in `configs/article.yaml`.

The environment file provides a starting point for a conda installation:

```bash
conda env create -f environment.yml
conda activate canary-change-probe
```

Install the package in editable mode when working from a checkout:

```bash
python -m pip install -e '.[test]'
```

## Configuration

Copy `configs/article.yaml` to a local file and replace the placeholder paths for:

1. the LEVIR-CD, WHU-CD, DSIFN-CD, and CLCD datasets;
2. the model checkpoints used by the article; and
3. the upstream model repositories required by the selected adapters.

Do not commit local data, checkpoints, or generated results. The configuration keeps the source-calibration and target runs explicit, and the pipeline writes results below `paths.results_root`.

## Run the checks

The validation command parses the article configuration without loading a model:

```bash
python scripts/run_article_audit.py --validate-only --config configs/article.yaml
```

The focused test suite does not require a GPU, dataset, checkpoint, or upstream model repository:

```bash
python -m pytest -q
```

## Run the article audit

After editing a local configuration, run a small CPU smoke run first:

```bash
python scripts/run_article_audit.py \
  --config configs/local.yaml \
  --model changemamba \
  --source levir_val=levir-256:val \
  --run levir_test=levir-256:test \
  --limit 4 \
  --skip-analyses
```

For the full configured set of target domains, omit `--run` and `--limit`. Use `--check-inputs` before a run to fail early when a dataset, checkpoint, or upstream repository is missing:

```bash
python scripts/run_article_audit.py \
  --check-inputs \
  --config configs/local.yaml
```

The pipeline stores probability maps and CSV files below `results/`. The main per-run files are `passive_scores.csv`, `canary_scores.csv`, and, when ground truth is available for offline evaluation, `eval_labels.csv`. The label-free run does not create or consume target labels for its decisions.

After the source and target runs finish, compute the source-normalized ratios and fixed F/I/NF decisions from the generated CSV files:

```bash
python scripts/summarize_audit.py \
  --results-root results \
  --model changemamba \
  --source levir_val=levir-256:val \
  --target whu_test=whu-256:test \
  --target clcd_test=clcd-256:test \
  --output results/article_audit.json
```

The summarizer reports the source means, target means, source-normalized (c) and (e) ratios, and the fixed primary rule decision. It does not reveal or require target masks.

## Interpretation and limitations

F means that at least one prespecified behavioral requirement was violated. I marks an intermediate result that requires inspection. NF means that the finite tests found no violation; it is not a target-recall estimate or a safety certificate. The article also reports that the audit ranking can match direct output silence in matched-coverage comparisons. Its additional value is structured behavioral diagnosis, not a claim of superiority over that simpler baseline.

Synthetic canaries are controlled probes rather than complete simulations of natural change. Evidence coverage is based on image-level differences and can be affected by illumination, seasonality, shadows, and misregistration. Interpret results within the evaluated datasets and model protocols.

## Repository layout

```text
configs/article.yaml        Example article configuration
scripts/run_article_audit.py  Validation and run entry point
scripts/run_pipeline.py     Article pipeline implementation
scripts/summarize_audit.py  Source-normalized ratios and F/I/NF decisions
src/scp/                    Article audit package
tests/                      Deterministic smoke tests
environment.yml             Conda environment specification
RELEASE_MANIFEST.md         Included and excluded content
```
