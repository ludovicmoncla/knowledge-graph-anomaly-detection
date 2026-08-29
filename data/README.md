# Data

This directory separates datasets and generated artifacts from the source code.

## `icews18/`

ICEWS18 files used by the BERT + R-GCN training pipeline:

- `entity2id.txt`: mapping from entity labels to integer identifiers;
- `relation2id.txt`: mapping from relation labels to integer identifiers;
- `train.txt`, `valid.txt`, and `test.txt`: temporal knowledge-graph splits.

The current pipeline reads `train.txt` and creates train, validation, and test
subsets independently for each temporal snapshot. The original `valid.txt` and
`test.txt` are retained for future experiments using the official dataset splits.

Run `bert-rgcn-describe-data` from the repository root to validate all three files and
generate readable tables, descriptive statistics, and the degree-distribution
figure under `data/processed/icews18/`. These derived files are reproducible and
therefore excluded from version control.

## `snapshots/`

Serialized graph snapshots created during the internship. Their schema,
provenance, and relationship to ICEWS18 still need to be documented before they
are used by the public pipeline.

## Publication note

Before publishing this repository, verify the redistribution terms and add the
dataset citation. If the data are hosted on Hugging Face instead, keep this file
and replace the large files with download instructions and stable dataset version
identifiers.
