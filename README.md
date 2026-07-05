# Diff-LNR

Diff-LNR predicts deletion pathogenicity from local DNA sequence alone. It
represents a deletion with a length-preserving counterfactual allele,
encodes the reference and counterfactual with a shared DNABERT-2 backbone, and
classifies the pooled-embedding difference.

This directory is the publication-ready implementation reconstructed from the
supplied research scripts. The scientific algorithm is unchanged:

1. Reference input: `L + D + R`.
2. Aligned counterfactual: `L + N^k + R`, where `k = len(REF) - len(ALT)`.
3. Shared DNABERT-2 encoder and pooler.
4. Difference vector: `h_ref - h_N`.
5. Classifier: `768 -> 256 -> ReLU -> dropout(0.1) -> 2`.
6. Train token embeddings, encoder blocks 7--12, and the classifier; freeze
   encoder blocks 1--6 and the pooler.
7. Optimize two-logit cross-entropy with AdamW.

## Repository structure

```text
diff_lnr_repo/
├── config.yaml                  # single source of experiment parameters
├── train.py                     # one-command holdout training
├── eval.py                      # checkpoint evaluation
├── cross_validate.py            # stratified k-fold evaluation
├── data/
│   └── README.md                # input schema and data placement
├── scripts/
│   ├── build_contexts.py
│   ├── create_length_matched_subset.py
│   ├── evaluate_regions.py
│   └── integrated_gradients.py
├── src/diff_lnr/
│   ├── config.py
│   ├── data/                    # records, sequence construction, splits, sampling
│   ├── layers/                  # shared genomic sequence encoder
│   ├── models/                  # classifier, backbone loading, layer freezing
│   ├── losses/                  # cross-entropy objective
│   ├── evaluation/              # AUROC, AUPRC, MCC, F1 and confusion matrix
│   ├── training/                # one training/evaluation implementation
│   ├── interpretability/        # branchwise Integrated Gradients
│   ├── workflows/               # end-to-end experiment assembly
│   └── utils/                   # logging, serialization, environment and seeds
├── tests/
├── docs/
├── pyproject.toml
└── requirements.txt
```

## Installation

Python 3.10 or later is required. A CUDA-enabled PyTorch installation is
recommended for training.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

For Integrated Gradients and development checks:

```bash
python -m pip install -e ".[interpretability,dev]"
```

DNABERT-2 uses `trust_remote_code=True`. Review the upstream model repository
before running it in a sensitive environment.

## Data preparation

Place a processed context file at the path configured by
`data.context_file`. The default is:

```text
data/clinvar_deletions_context_400.txt
```

See [data/README.md](data/README.md) for the seven-column schema. To regenerate
context windows from a reference FASTA:

```bash
python scripts/build_contexts.py \
  --variants data/clinvar_deletions_context_template.txt \
  --fasta /path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
  --output-dir data \
  --flank 150 200 250 300
```

The command verifies each REF allele against the FASTA and reports
out-of-bounds and reference-mismatch counts.

## Validate before training

This checks the configuration, input schema, filtering, labels, class counts,
and first reconstructed record without downloading the model:

```bash
python train.py --config config.yaml --validate-only
```

A small schema fixture is included for an immediate installation check:

```bash
python train.py --config configs/smoke.yaml --validate-only
```

## One-command training

```bash
python train.py --config config.yaml
```

Useful overrides:

```bash
python train.py --config config.yaml \
  --data data/clinvar_deletions_context_400.txt \
  --model zhihan1996/DNABERT-2-117M \
  --output-root runs \
  --run-name diff_lnr_400bp_seed42 \
  --device cuda
```

Each run saves:

```text
runs/<run_name>/
├── config.resolved.yaml
├── environment.json
├── data_loading.json
├── splits.json
├── split_summary.json
├── trainable_parameters.json
├── run.log
├── history.csv
├── history.json
├── best_metrics.json
├── checkpoints/best.pt
├── tokenizer/
├── test/metrics.json
├── test/predictions.csv
└── summary.json
```

The saved split indices and tokenizer make the test evaluation auditable.

## Evaluation

```bash
python eval.py \
  --config runs/<run_name>/config.resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/best.pt \
  --splits runs/<run_name>/splits.json
```

## Five-fold cross-validation

```bash
python cross_validate.py --config config.yaml --folds 5
```

Every fold receives a fresh backbone and classifier, a fold-specific validation
split, checkpoint, prediction file, and saved indices. The final
`cv_summary.json` contains mean, sample standard deviation, and individual
values for accuracy, precision, recall, F1, AUROC, AUPRC, MCC, and loss.

## Reproduce the six sequence representations

Set `model.representation` in `config.yaml`:

| Value | Input |
|---|---|
| `ldr` | `L + D + R` |
| `lr` | `L + R` |
| `lnr` | `L + N^k + R` |
| `diff_lr` | `E(L+D+R) - E(L+R)` |
| `diff_lnr` | `E(L+D+R) - E(L+N^k+R)` |
| `d_only` | deleted bases `D` |

All representations use the same backbone, classifier, optimizer, split, and
freezing policy. This isolates input construction from model capacity.

## Robustness and interpretability

Length matching:

```bash
python scripts/create_length_matched_subset.py \
  --config config.yaml \
  --max-difference 5 \
  --output data/length_matched_indices.json
```

Set `data.subset_indices_file` to the generated file, then run training or
cross-validation normally.

Coding/noncoding evaluation:

```bash
python scripts/evaluate_regions.py \
  --config runs/<run_name>/config.resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/best.pt \
  --splits runs/<run_name>/splits.json \
  --vcf /path/to/clinvar.vcf.gz
```

By default, positions absent from the VCF are assigned to the noncoding group,
matching the supplied control script. Use `--keep-unmatched-separate` for a
more conservative audit.

Integrated Gradients:

```bash
python scripts/integrated_gradients.py \
  --config runs/<run_name>/config.resolved.yaml \
  --checkpoint runs/<run_name>/checkpoints/best.pt \
  --splits runs/<run_name>/splits.json \
  --confidence 0.95 \
  --samples 200 \
  --steps 32 \
  --window 200 \
  --convergence-steps 16 32 64 128
```

The analysis attributes the reference and aligned counterfactual separately,
maps token attributions back to base coordinates, and saves their normalized
absolute difference around the deleted-interval center.

## Reproducibility controls

- Python, NumPy and PyTorch seeds are fixed.
- CUDA seeds are fixed when CUDA is available.
- Deterministic PyTorch algorithms are requested with warning-only fallback.
- Length-bucket shuffling uses `seed + epoch`.
- Train/validation/test indices are saved.
- Cross-validation folds are saved.
- The exact resolved configuration and package versions are saved.
- The best validation-F1 checkpoint is restored before test evaluation.
- Per-record probabilities are exported.

## Important provenance limitations

The supplied source directory did **not** contain:

1. the `baselines` package imported by the original CNN, BiLSTM, and XGBoost
   scripts;
2. the local `NADDED` model/tokenizer directory;
3. the processed ClinVar context files;
4. trained checkpoints or raw experiment JSON files.

The anonymous public repository contained only a README when checked during
this refactor. Consequently, this repository does not guess the missing
baseline architectures or claim byte-for-byte reproduction of the reported
metrics. The core Diff-LNR implementation is runnable after data and model
dependencies are supplied. See
[docs/ENGINEERING_AUDIT.md](docs/ENGINEERING_AUDIT.md).

The supplied loader retained `Likely_benign` and `Likely_pathogenic` labels
while excluding uncertain/conflicting labels. This behavior remains the
default (`skip_likely_labels: false`) to avoid silently changing the original
algorithm. Set it to `true` only when intentionally reproducing a curation
protocol that excludes likely labels.

## Testing

```bash
python -m pytest
python -m compileall -q src train.py eval.py cross_validate.py scripts
```

Before publishing results, complete the checklist in
[docs/REPRODUCIBILITY_CHECKLIST.md](docs/REPRODUCIBILITY_CHECKLIST.md).
