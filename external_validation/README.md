# Chapter 5 external validation: MEISD → SemEval 2018 EI-oc

This self-contained experiment tests a narrow, defensible transfer claim: does
the *intensity encoder* learned with Chapter 5's soft-sharing multi-task setup
transfer better to an independent emotion-intensity dataset than (a) a fresh
BERT encoder or (b) an encoder trained on the same MEISD rows for intensity
alone? It does **not** test dialogue-level forecasting or transfer of the
Chapter 5 output heads. Its results belong in a short external-validation
subsection of Chapter 5, followed by a limitation in the discussion.

The target is the English **EI-oc** task of [SemEval-2018 Task 1](https://aclanthology.org/S18-1001/):
four emotion-conditioned ordinal classes (0–3) for anger, fear, joy, and
sadness. The official train/development/test files are used without
resplitting. A separate four-class target head is initialized afresh for each
arm and each seed. The encoder and target head are then fine-tuned together
on *target training data only*. Checkpoints are selected using target
development macro Pearson; test labels do not enter fitting or selection. The
main number is the unweighted mean of Pearson correlations for the four
emotions; per-emotion Pearson, macro-F1, and MAE are also saved.

## Conditions

| Arm | Target encoder initialization | Source training |
| --- | --- | --- |
| `target_only` | `bert-base-cased` | None |
| `meisd_stl` | MEISD intensity encoder | Intensity task only |
| `meisd_soft_mtl` | MEISD intensity encoder | Intensity + emotion + sentiment, three BERT encoders with a soft parameter-sharing penalty |

The two source encoders use the **same raw MEISD rows**, dialogue-level
source train/development split, source seed, base model, batch size, and
optimization settings. The MTL objective weights intensity/emotion/sentiment
losses by 0.7/2.0/1.0 and uses the same pairwise encoder L2 penalty as the
Chapter 5 implementation. This is a clean retraining on raw MEISD, rather
than a reuse of the published augmented-data checkpoint. The intensity
encoder is selected by MEISD development intensity macro-F1. Ambiguous or
missing intensity labels are excluded, and exact normalized source-text
overlap with SemEval is removed. `--check-data` reports all exclusions and
file hashes.

## Run

Use Python 3.10 or newer and PyTorch 2.2 or newer. Install a suitable [PyTorch build](https://pytorch.org/get-started/locally/)
for the available GPU, then install Transformers:

```powershell
python -m pip install "transformers>=4.40,<5"
python -m external_validation.semeval_transfer --download --check-data
python -m external_validation.semeval_transfer --download
```

Run from the repository root. `--download` fetches the official data archive
from the task author's site into `data/`; it is ignored by Git. If the archive
is already there, the command uses that copy. Use `--device cpu` if CUDA is
unavailable, but expect full three-encoder source training to be slow.
`python -m external_validation.semeval_transfer --help` lists the optional
hyperparameters. The defaults run three paired target seeds (42, 43, 44),
three source epochs, four target epochs, batch size 8, and BERT cased.
Use identical target settings for every arm. For a quick pipeline smoke test,
you may reduce epochs and seeds, but do not report such a run as the primary
validation.

The script writes `outputs_semeval_transfer/run_manifest.json` (dataset
hashes, settings, versions), source encoder checkpoints, all target
checkpoints, `results.json` (each seed and emotion), `summary.json` (mean,
standard deviation, paired differences), `summary_table.csv` (ready-to-read
three-arm table), and `test_predictions.csv`. Output
and downloaded data are Git-ignored. If source training is interrupted,
rerunning with the same settings reuses a completed source encoder; changing
source settings requires a new `--output-dir`.

There is an optional `--historical-mtl-checkpoint PATH
--allow-nonmatched-source` shortcut for the old Chapter 5 checkpoint. That
checkpoint was trained with a different, augmented source table, so this
mode is explicitly **exploratory** and should not be used for the matched
three-arm comparison. The default command above trains both source arms on
matched raw MEISD data.

## Interpretation and thesis reporting

Report the three-arm means and standard deviations and paired seed-wise
differences (`meisd_soft_mtl` minus both comparators). Include all four
per-emotion results, even if some decline. A positive paired difference over
`meisd_stl` supports a contribution of multi-task source learning beyond
generic MEISD intensity pretraining; a positive difference over
`target_only` supports transfer from MEISD. If neither holds consistently,
report the negative result and narrow the transfer claim. This single target
dataset is an *external check*, not evidence of universal generalization.

Suggested compact Chapter 5 placement: one subsection with a paragraph of
protocol, a three-row results table (plus four emotion columns), and one
paragraph interpreting paired seed results. Discuss different domains and
annotation schemes, and the small number of seeds, as limitations. Do not
merge these SemEval scores with the original MEISD or ESConv tables: the
target task has different labels and evaluation units.
