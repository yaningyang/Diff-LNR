# Engineering audit

## Scope

The audit covered every Python file supplied in the original `代码/` directory
and the accompanying ZIP archive. The clean implementation was derived from
the executable sequence construction, model forward pass, layer-freezing
policy, optimization loop, cross-validation code, length matching, region
control, and Integrated Gradients scripts.

## Before / after

| Area | Before | After |
|---|---|---|
| Configuration | Parameters duplicated across more than ten scripts | One typed YAML configuration with validation |
| Data paths | Hard-coded local Windows paths | Paths resolved relative to the config file; CLI overrides supported |
| Data flow | Parsing, sequence reconstruction and region positions duplicated | One `DeletionRecord` parser retaining coordinates and provenance |
| Representations | String combinations spread across `main.py` and `data.py` | One explicit six-representation mapping |
| Model | Legacy unused feature-embedding branch and commented code | One shared encoder plus the paper's classifier |
| Fine-tuning | Parameter freezing embedded inside the training loop | Named, testable selective fine-tuning function |
| Training | Separate loops for main model and multiple experiment scripts | One engine for training, validation and evaluation |
| Metrics | Different scripts reported different metric sets | Accuracy, precision, recall, F1, AUROC, AUPRC and MCC everywhere |
| Cross-validation | Fold training artifacts could overwrite a shared run directory | Isolated fold directories with saved indices and checkpoints |
| Randomness | Only scikit-learn split seeds were explicit | Python, NumPy, PyTorch, CUDA and sampler seeds are controlled |
| Results | Logs, checkpoints and metrics saved inconsistently | Stable run directory schema with predictions and environment manifest |
| Documentation | No user README or input schema | Installation, data format, commands, controls and limitations documented |
| Comments | Many source comments were corrupted by character encoding | Concise English docstrings and comments |

## Preserved algorithmic decisions

- Reference reconstruction from the alternate context TSV.
- Deleted sequence defined as `REF[len(ALT):]`.
- N placeholders inserted immediately after the retained ALT anchor.
- Shared DNABERT-2 weights for both sequence branches.
- Reference embedding minus alternate/counterfactual embedding.
- Pooler output by default.
- `768 -> 256 -> 2` classifier with ReLU and dropout 0.1.
- Train full token embeddings, encoder layers 6--11, and classifier.
- Freeze lower encoder layers and pooler.
- AdamW, learning rate `5e-5`, weight decay `0.05`.
- Two-logit cross-entropy and validation-F1 early stopping.
- Deletion-length bucketed training batches.
- 80/10/10 stratified holdout and stratified five-fold evaluation.
- Greedy one-to-one length matching within 5 bp.
- Coding status derived from ClinVar `MC` consequences.
- Separate reference/alternate Integrated Gradients followed by base-space
  absolute difference and per-sample max normalization.

## Findings that could not be repaired without inventing evidence

### Missing baseline package

`run_baseline.py` and `run_baseline_length_comparison.py` import:

- `baselines.baseline_dataset`
- `baselines.cnn_baseline`
- `baselines.bilstm_baseline`
- `baselines.kmer_xgboost`

None of these files was present locally, in the supplied ZIP, or in the
anonymous repository at audit time. Reconstructing them from class names would
not guarantee the architectures used for the paper and would violate the
requirement not to change the algorithm. They are therefore not fabricated.

### Missing exact backbone artifact

The experiment scripts reference a local `NADDED` directory that was not
supplied. The user-facing default loads the public DNABERT-2 checkpoint and
adds `N` to the tokenizer at runtime. This reproduces the documented method,
but exact numerical reproduction requires confirmation that `NADDED` contained
no additional learned changes.

### Label-curation conflict

The supplied loader maps `Likely_benign` to benign and
`Likely_pathogenic` to pathogenic. Some manuscript prose states that qualified
likely labels were excluded. The clean configuration exposes this choice
explicitly and preserves the supplied code's behavior by default.

### Missing data and outputs

No processed context TSV, raw cross-validation JSON, trained checkpoint, or IG
NPZ source was available. Static and synthetic tests can validate the software
contracts; reported paper numbers cannot be recomputed from this workspace.
