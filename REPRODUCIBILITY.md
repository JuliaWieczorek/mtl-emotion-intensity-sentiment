# Reproducing the joint emotion-intensity-sentiment study

## Scope

This repository is the publication-facing snapshot for the three-task MTL
manuscript. The related `meisd_project/pipeline/EMOTIA/` directory documents
earlier development but is not the canonical citation target.

## Pipeline map

1. Analyse and prepare MEISD with the scripts in `EMOTIA-DA/`.
2. Produce the multi-label-aware augmented training table.
3. Configure the required task set and architecture in
   `EMOTIA-ML/multi_emotion_sentiment_intensity_classifier.py`.
4. Run the single-task, pairwise-task, and three-task experiments separately.
5. Analyse task interaction and architecture comparisons with
   `EMOTIA-ML/analyse_multitask_learnig_anova_etc.py`.

## Historical entry point

```powershell
python EMOTIA-ML/multi_emotion_sentiment_intensity_classifier.py
```

The script expects a generated CSV at a project-relative historical location
and stores outputs in a configured directory. Confirm the resolved paths before
launching a long GPU run.

## Required run manifest

Record:

- Git commit;
- input dataset and augmentation-manifest hashes;
- architecture and transformer backbone;
- enabled task combination;
- task weights (`sentiment`, `emotion`, and `intensity`);
- loss functions and focal-loss parameters;
- seed, split definition, and epoch selected by early stopping;
- Python, PyTorch, Transformers, CUDA, and hardware versions;
- prediction, metric, and training-history hashes.

## Completion criteria

A reproduction must recreate the correct task-specific data counts and compare
single-task, two-task, and three-task results. Report sentiment, multi-label
emotion, and emotion-conditioned intensity metrics separately; aggregate loss
alone is not sufficient evidence of reproduction.

