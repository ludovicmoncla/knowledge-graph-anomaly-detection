# SERAD-KG

**Semantic and Relational Anomaly Detection in Knowledge Graphs**

SERAD-KG detects anomalous events in temporal knowledge graphs by combining two
complementary views of a triple:

- a **local semantic branch**, built from Sentence-Transformer embeddings;
- a **global relational branch**, based on an R-GCN encoder and a TransE score.

The repository also contains a reproducible implementation of **LoGNet**, used as
the baseline, as well as the complete data-preparation and repeated-evaluation
workflow used for ICEWS18.

The methodology and code were developed by **Antoine Salazar** during his
final-year internship at LIRIS, under the supervision of
[Ludovic Moncla](https://ludovicmoncla.github.io),
[Yassir Lairgi](https://lairgiyassir.github.io),
[Khalid Benabdeslem](http://kbenabde.free.fr/bk/index.php), and
[Rémy Cazabet](https://cazabetremy.fr).

![SERAD-KG architecture and experimental pipeline](docs/images/pipeline-overview.png)

## Main results

The following results use LLM-generated anomalies, five initialization seeds
(41–45), a 10% anomaly ratio in training, validation, and test, and `alpha = 0.5`
for SERAD-KG. Values are the **mean ± sample standard deviation** across seeds.
Anomalies are the positive class for AUPRC, precision, recall, and F1.

| Protocol | Method | AUROC | AUPRC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Pooled | LoGNet | 0.614 ± 0.020 | 0.143 ± 0.008 | 0.138 ± 0.012 | **0.492 ± 0.127** | 0.213 ± 0.013 |
| Pooled | **SERAD-KG** | **0.786 ± 0.003** | **0.321 ± 0.011** | **0.308 ± 0.016** | 0.451 ± 0.055 | **0.364 ± 0.012** |
| Chronological | LoGNet | 0.627 ± 0.021 | 0.150 ± 0.010 | 0.156 ± 0.022 | 0.412 ± 0.144 | 0.219 ± 0.010 |
| Chronological | **SERAD-KG** | **0.811 ± 0.004** | **0.359 ± 0.009** | **0.357 ± 0.020** | **0.423 ± 0.036** | **0.386 ± 0.011** |

SERAD-KG improves AUPRC by 0.178 in the pooled protocol and 0.209 in the
chronological protocol at this training ratio. LoGNet has slightly higher pooled
recall, but with substantially lower precision; threshold-independent AUROC and
AUPRC therefore provide the more informative comparison.

### Influence of the training anomaly ratio

Validation and test remain fixed at 10% anomalies in this experiment. Only the
training anomaly ratio changes. The table reports test AUPRC and F1.

| Protocol | Method | Metric | 1% | 2.5% | 5% | 10% | 20% |
|---|---|---|---:|---:|---:|---:|---:|
| Pooled | LoGNet | AUPRC | 0.123 ± 0.006 | 0.127 ± 0.006 | **0.162 ± 0.011** | 0.143 ± 0.008 | 0.115 ± 0.005 |
| Pooled | LoGNet | F1 | 0.191 ± 0.005 | 0.195 ± 0.005 | **0.244 ± 0.019** | 0.213 ± 0.013 | 0.184 ± 0.003 |
| Pooled | SERAD-KG | AUPRC | 0.231 ± 0.005 | 0.257 ± 0.004 | 0.292 ± 0.005 | 0.321 ± 0.011 | **0.382 ± 0.015** |
| Pooled | SERAD-KG | F1 | 0.300 ± 0.012 | 0.325 ± 0.007 | 0.345 ± 0.006 | 0.364 ± 0.012 | **0.405 ± 0.011** |
| Chronological | LoGNet | AUPRC | 0.119 ± 0.005 | 0.121 ± 0.005 | **0.172 ± 0.016** | 0.150 ± 0.010 | 0.110 ± 0.007 |
| Chronological | LoGNet | F1 | 0.179 ± 0.015 | 0.182 ± 0.014 | **0.244 ± 0.010** | 0.219 ± 0.010 | 0.180 ± 0.019 |
| Chronological | SERAD-KG | AUPRC | 0.233 ± 0.009 | 0.262 ± 0.009 | 0.291 ± 0.010 | 0.359 ± 0.009 | **0.413 ± 0.016** |
| Chronological | SERAD-KG | F1 | 0.291 ± 0.006 | 0.301 ± 0.012 | 0.332 ± 0.023 | 0.386 ± 0.011 | **0.435 ± 0.011** |

SERAD-KG benefits consistently from additional labeled anomalies, whereas LoGNet
peaks at 5% in both protocols. These are experimental observations, not a ratio
selected independently on validation data. Detailed tables, plots, validation-based
selection, and paired confidence intervals are available in
[`notebooks/llm_ratio_results.ipynb`](notebooks/llm_ratio_results.ipynb).

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For tests and development tools:

```bash
python -m pip install -e '.[dev]'
```

## Data

The commands expect an ICEWS18 directory with the following files:

```text
data/icews18/
├── entity2id.txt
├── relation2id.txt
├── train.txt
├── valid.txt
└── test.txt
```

The mapping files contain a label and an integer identifier separated by a tab.
Each split contains subject, relation, object, and timestamp identifiers, plus an
optional fifth column. See [`data/README.md`](data/README.md) for the detailed data
layout and licensing notes.

To validate and summarize the dataset:

```bash
serad-kg-describe-data \
  --data-dir data/icews18 \
  --output-dir data/processed/icews18
```

## Reproducing the LLM-anomaly experiments

### 1. Prepare shared manifests

Preparation is deliberately separated from training. It creates immutable
`manifest.csv` files consumed by both methods. Their SHA-256 checksums are copied
to the run configurations so that SERAD-KG and LoGNet can be verified to use the
same examples and splits.

The pooled experiment combines the first 30 snapshots, deduplicates triples, and
then performs a stratified 60/20/20 split:

```bash
serad-kg-prepare \
  --data-dir data/icews18 \
  --output-dir data/processed/icews18/experiments/pooled_30_llm \
  --protocol pooled \
  --negative-sampling cache \
  --anomaly-cache-dir data/processed/icews18/llm_anomalies_30 \
  --num-snapshots 30 \
  --max-anomaly-ratio 0.20 \
  --evaluation-anomaly-ratio 0.10 \
  --train-anomaly-ratios 0.01 0.025 0.05 0.10
```

The chronological experiment uses snapshots 0–6 for training, snapshot 7 for
validation, and snapshots 8–9 for testing:

```bash
serad-kg-prepare \
  --data-dir data/icews18 \
  --output-dir data/processed/icews18/experiments/chronological_llm \
  --protocol chronological \
  --negative-sampling cache \
  --anomaly-cache-dir data/processed/icews18/llm_anomalies_30 \
  --num-snapshots 10 \
  --max-anomaly-ratio 0.20 \
  --evaluation-anomaly-ratio 0.10 \
  --train-anomaly-ratios 0.01 0.025 0.05 0.10
```

An anomaly ratio is defined as `anomalies / (normal examples + anomalies)`. The
root manifest uses 20% training anomalies. Nested `train_ratio_*` directories use
1%, 2.5%, 5%, and 10%, while sharing the same normal training examples and the
same validation and test sets.

To regenerate anomalies through OpenRouter instead of using the cache, replace
`cache` with `genai` and add:

```text
--openrouter-model openai/gpt-4.1 --genai-batch-size 50
```

Set `OPENROUTER_API_KEY` in `.env`. Generation can incur API costs. If it is
interrupted, rerun the same command with `--resume-genai`; accepted batches are
checkpointed and reused.

### 2. Run the repeated experiments

The reproducible CPU scripts use seeds 41–45, 300 maximum epochs, patience 25,
dynamic score normalization, no edge masking, no ranking loss, and no model
checkpoints. Metrics, per-example scores, histories, plots, configurations, and
execution times are retained.

```bash
# SERAD-KG, alpha = 0.5, training ratios 1%, 2.5%, 5%, and 20%
./scripts/run_serad_kg_llm_ratios_cpu.sh

# LoGNet, training ratios 1%, 2.5%, 5%, and 20%
./scripts/run_lognet_llm_ratios_cpu.sh
```

The 10% runs are separate because they were also used for the alpha study:

```bash
# SERAD-KG, alpha values 0.5, 0.75, 0.85, 0.95, and 1.0
./scripts/run_serad_kg_llm_cpu.sh

# LoGNet
./scripts/run_lognet_llm_cpu.sh
```

The ratio-sweep scripts support resumption: completed seeds are skipped with
`--resume`. On macOS, `caffeinate -i` can keep the machine awake during a long run:

```bash
caffeinate -i ./scripts/run_serad_kg_llm_ratios_cpu.sh
```

## Running one configuration

SERAD-KG example:

```bash
serad-kg-train \
  --data-dir data/icews18 \
  --prepared-data-dir data/processed/icews18/experiments/pooled_30_llm/train_ratio_0.1 \
  --output-dir outputs/example/serad_kg \
  --alpha 0.5 \
  --epochs 300 \
  --patience 25 \
  --score-normalization dynamic \
  --edge-mask-ratio 0 \
  --ranking-loss-weight 0 \
  --device cpu \
  --no-save-model \
  --seeds 41 42 43 44 45
```

LoGNet example using the exact same prepared data:

```bash
lognet-train \
  --data-dir data/icews18 \
  --prepared-data-dir data/processed/icews18/experiments/pooled_30_llm/train_ratio_0.1 \
  --output-dir outputs/example/lognet \
  --epochs 300 \
  --patience 25 \
  --device cpu \
  --no-save-model \
  --seeds 41 42 43 44 45
```

Pass `--resume` to preserve completed seeds when restarting a repeated run.

## Model and training

For a local plausibility logit `s_local` and a global relational logit `s_global`,
the combined representation is controlled by `alpha`:

```text
combined = alpha × calibrated(s_local) + (1 - alpha) × calibrated(s_global)
```

Thus `alpha = 1` uses only the semantic branch, `alpha = 0` only the relational
branch, and intermediate values combine both. Each branch is standardized using
training-only statistics and receives a learned positive scale and bias.

The main SERAD-KG configuration uses:

- class-balanced BCE on the combined score;
- an auxiliary BCE loss on each branch (weight 0.25);
- dimension-normalized TransE distances;
- validation AUPRC for learning-rate scheduling, checkpoint selection, and early
  stopping;
- a classification threshold selected exclusively on validation data by maximizing
  anomaly-class F1.

A differentiable pairwise ranking loss and positive-edge masking are implemented
for ablations but disabled in the reported experiments. They can be enabled with
`--ranking-loss-weight`, `--ranking-margin`, and `--edge-mask-ratio`.

## Evaluation and outputs

No test label is used for optimization, early stopping, or threshold selection.
Final precision, recall, F1, AUROC, and AUPRC are computed once on the test set.
AUROC and AUPRC use the continuous anomaly score and do not depend on the selected
classification threshold.

A repeated run produces:

```text
outputs/<experiment>/
├── seed_41/
│   ├── config.json
│   ├── summary.csv
│   ├── scores.csv
│   └── training_history.csv
├── seed_42/
│   └── ...
├── summary_by_seed.csv
├── summary_aggregate.csv
└── overall_aggregate.csv
```

Depending on the protocol, plots and snapshot-specific subdirectories can also be
created. `model.pt` is written unless `--no-save-model` is supplied. The experiment
scripts disable checkpoint storage to limit disk usage while preserving all scores
and timings (`elapsed_seconds`).

## Reproducibility notes

- Use the same prepared-data directory for both methods.
- Compare runs on the same device; CPU was used for the reported tables.
- Keep all five seeds for final comparisons. A single seed is suitable only for
  preliminary debugging.
- Pooled and chronological results answer different questions and should not be
  merged: pooled measures performance on a random split, while chronological
  measures generalization to future snapshots.
- LoGNet and SERAD-KG retain their method-specific optimization procedures. The
  current comparison therefore evaluates the complete methods, not only their
  architectures.

## Tests

```bash
pytest
```

## References

- Michael Schlichtkrull et al., [*Modeling Relational Data with Graph Convolutional
  Networks*](https://arxiv.org/abs/1703.06103), ESWC 2018.
- Antoine Bordes et al., [*Translating Embeddings for Modeling Multi-relational
  Data*](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html),
  NeurIPS 2013.
- Alberto García-Durán, Sebastijan Dumančić, and Mathias Niepert,
  [*Learning Sequence Encoders for Temporal Knowledge Graph
  Completion*](https://aclanthology.org/D18-1516/), EMNLP 2018.
- Nils Reimers and Iryna Gurevych,
  [*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*](https://aclanthology.org/D19-1410/),
  EMNLP-IJCNLP 2019.

## License

The source code is distributed under the terms of [`LICENSE`](LICENSE). Dataset,
LLM-generated data, and pretrained-model licenses apply independently.
