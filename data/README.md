# Data

Place the processed context TSV files in this directory. Data files are not
committed because ClinVar and the reference genome must be distributed under
their own terms.

Expected columns:

| Column | Meaning |
|---|---|
| `CHROM` | chromosome name |
| `POS` | 1-based VCF position |
| `REF` | reference allele, including the retained VCF anchor |
| `ALT` | alternate allele, usually the retained anchor |
| `labels` | ClinVar significance string |
| `context_sequence` | alternate window: left flank + ALT + right flank |
| `var_position` | zero-based position of ALT in `context_sequence` |

The loader reconstructs the reference window as:

```text
context[:var_position] + REF + context[var_position + len(ALT):]
```

For Diff-LNR, the aligned counterfactual inserts
`N * (len(REF) - len(ALT))` immediately after the retained ALT anchor.

The manuscript experiments use context files corresponding to 300, 400, 500,
and 600 bp flank settings. Set `data.context_file` in `config.yaml` to the
desired file.
