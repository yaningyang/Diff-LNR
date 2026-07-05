# Reproducibility checklist

## Before running

- [ ] Confirm Python and CUDA versions.
- [ ] Record the DNABERT-2 model revision or local checkpoint checksum.
- [ ] Record the ClinVar release date and genome build.
- [ ] Confirm the reference FASTA checksum.
- [ ] Verify REF alleles during context extraction.
- [ ] Decide whether likely labels are included.
- [ ] Confirm the maximum REF allele length.
- [ ] Run `python train.py --config config.yaml --validate-only`.

## Training

- [ ] Keep `seed: 42` for paper reproduction.
- [ ] Confirm representation is `diff_lnr`.
- [ ] Confirm pooling is `pooler`.
- [ ] Confirm trainable encoder layers are `[6, 11]`.
- [ ] Inspect `trainable_parameters.json`.
- [ ] Preserve `config.resolved.yaml`, `environment.json`, and `splits.json`.
- [ ] Retain `run.log`, `history.csv`, and `checkpoints/best.pt`.

## Evaluation

- [ ] Evaluate only after restoring the best validation-F1 checkpoint.
- [ ] Report class counts for every split.
- [ ] Report AUROC, AUPRC, MCC, F1 and confusion matrix.
- [ ] Archive per-record predictions.
- [ ] For cross-validation, report individual folds and sample standard deviation.
- [ ] Do not mix single-split ablations with fold-averaged baseline results.

## Controls

- [ ] Save length-matched subset indices and matching statistics.
- [ ] Report VCF-unmatched positions in the coding/noncoding control.
- [ ] Save IG integration steps, threshold, selected record IDs and NPZ profile.
- [ ] Run IG convergence at 16, 32, 64 and 128 steps.

## Before release

- [ ] Run `python -m pytest`.
- [ ] Run `python -m compileall -q src train.py eval.py cross_validate.py scripts`.
- [ ] Remove local data, model weights and identifying paths.
- [ ] Verify all README commands in a clean environment.
- [ ] Add the missing baseline source or explicitly retain the limitation.
- [ ] Tag the exact commit used for the manuscript.
