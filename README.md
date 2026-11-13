# Motion ID

Human activity recognition from smartphone motion sensors.

Motion ID is an undergraduate artificial intelligence research project investigating whether machine learning models can identify six everyday activities from a smartphone worn at the waist. The study compares Support Vector Machines (SVM) and Random Forest on the UCI Human Activity Recognition Using Smartphones dataset, with evaluation on participants absent from training.

## Status

Research in progress. The data loader, participant-grouped validation, and evaluation metrics are implemented and tested. The SVM and Random Forest models are not implemented yet, and no experimental results exist. This project uses numerical sensor measurements, not images or videos.

## Research objectives

- Measure recognition performance on unseen participants.
- Compare SVM and Random Forest in predictive quality, training time, and prediction time.
- Identify commonly confused activities and investigate relevant sensor features.
- Produce reproducible experiments and an academic paper using an IEEE template, subject to course confirmation.

## Research questions

1. How accurately can the six activities be recognized for participants not seen during training?
2. How do SVM and Random Forest compare in recognition performance and computational cost?
3. Which activities are most frequently confused, and which sensor features help distinguish them?

## Dataset

The [UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) contains recordings from 30 volunteers carrying a Samsung Galaxy S II at the waist. Accelerometer and gyroscope signals were sampled at 50 Hz and divided into 2.56-second windows with 50% overlap.

| Property | Value |
| --- | --- |
| Total windows | 10,299 |
| Features per window | 561 |
| Official training set | 7,352 windows from 21 participants |
| Official test set | 2,947 windows from 9 different participants |
| Activities | Walking, walking upstairs, walking downstairs, sitting, standing, lying down |

The original archive includes windowed sensor signals, calculated features, activity labels, and participant identifiers. Rows in the corresponding files refer to the same window. Participant identifiers are used for splitting and validation, never as model input.

Download the original archive from UCI and extract it to `data/raw/UCI HAR Dataset/`. See [dataset documentation](docs/dataset.md) for file names and checks. Raw data and trained models are excluded from Git.

## Planned methodology

Preserve the official training and test sets. Compare both models using the same participant-based validation folds within the training set. Fit any preprocessing only on each training fold, using a pipeline. Freeze model settings before the final test evaluation.

Report accuracy, macro precision, macro recall, macro F1, confusion matrices, and processing times. Record parameters, software versions, hardware, and the code revision used. The number of folds, search ranges, primary selection metric, and timing procedure remain to be agreed before experiments begin.

Logistic Regression is an optional baseline. Additional models are secondary to completing the two main comparisons.

## Repository layout

```text
motion-id/
  README.md
  pyproject.toml         # dependencies and pytest settings, managed with uv
  requirements.txt       # pinned scikit-learn version
  configs/
    default.yaml         # folds, seed, and selection metric
  src/har/
    data.py              # loads the UCI HAR files (features, labels, participants)
    config.py            # reads configs/default.yaml
    validation.py        # participant-grouped folds and overlap checks
    metrics.py           # accuracy, macro precision/recall/F1, confusion matrix
    evaluate.py          # cross-validation, final test run, JSON and CSV reports
  tests/                 # one test file per module
  data/
    README.md
    raw/                 # local original data, ignored
  docs/
    proposal.md          # first-cycle research proposal
    dataset.md           # data layout, provenance, and limitations
    proposal.pdf         # Portuguese submission document
```

## Setup and tests

```bash
uv sync
uv run pytest
```

Tests that use the real dataset are skipped if it is not in `data/raw/UCI HAR Dataset/`. Notebooks, model code, and result directories will be added when implemented. There are no training commands yet.

## Research documents

- [Research proposal](docs/proposal.md)
- [Submission PDF in Portuguese](docs/Motion-ID_Artigo_IEEE_.pdf)
- [Dataset and references](docs/dataset.md)

## Limitations

The dataset represents controlled activities, one smartphone model, and waist placement. Results will not establish performance for other device positions, phones, populations, or everyday environments. Adjacent windows overlap, so random row splitting must not replace participant-based validation.

## Collaboration

Keep `main` as the reviewed project version and use `dev` for integration. Create short task branches from `dev` and use Conventional Commits. Review changes before integration, record actual checks, and keep documentation aligned with implemented behavior. Never commit credentials, raw datasets, or large model files. Do not report planned experiments as completed results.

## Citation and licensing

Dataset: J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, and X. Parra, *Human Activity Recognition Using Smartphones*, UCI Machine Learning Repository, 2013. DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).

The current UCI listing specifies CC BY 4.0 for the dataset. The project code license has not been selected.
