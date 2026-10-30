# Research execution plan

## Scope and status

This English working plan summarizes the supplied planning document. The original Word document and course brief are preserved locally in `docs/sources/` and excluded from Git. This repository contains planning documentation only; no experiment is complete.

Required work: inspect the data; compare SVM and Random Forest; report predictive metrics, confusion matrices, and timings; analyze errors and features; preserve experimental settings; and produce reproducible code and an academic paper.

Optional work: Logistic Regression as a baseline, followed by KNN only after required work is complete. Real-time applications, cameras, fall detection, new data collection, and deep model training are outside the current scope.

## Validation protocol

- Preserve the official training and test sets.
- Verify that participants do not overlap between those sets or between validation folds.
- Use GroupKFold with participant IDs inside the official training set.
- Fit preprocessing only within the training portion of each fold.
- Select settings using validation results and freeze them before testing.
- Reserve the test set for final evaluation. Repeat only the same frozen procedure for reproducibility, without further tuning.
- Record accuracy, macro precision, macro recall, macro F1, confusion matrices, search time, final training time, and prediction time.
- Record seed 42 where applicable, package versions, hardware, and code revision. Generate tables and figures from saved results.

Before experiments, agree on fold count, search ranges, the primary selection metric, timing repetitions, parallelism, and a feature interpretation procedure that addresses specific activity confusions.

## Team responsibilities

| Owner | Main responsibility | Writing responsibility |
| --- | --- | --- |
| Member 1 | Data loading, exploration, data checks, initial repository documentation | Introduction and dataset |
| Member 2 | Participant validation, metrics, optional baseline, comparison table | Methodology and comparison |
| Member 3 | SVM, environment checks, confusion analysis | SVM and error analysis |
| Member 4 | Random Forest, feature analysis, figure consistency | Random Forest and feature interpretation |

Each member contributes tests for their code, reviews another member's work, and writes and reviews paper sections. Reassign blocked work promptly. The original estimate of 10 to 13 technical hours per member is provisional.

## Milestones

The supplied plan assumes a final deadline of November 13, 2026. Confirm the date in Canvas, along with paper language, IEEE format, page limit, and repository access requirements.

| IDs | Work | Target in supplied plan |
| --- | --- | --- |
| T00-T01 | Repository, access, course requirements, dataset inspection | October 30-31 |
| T02-T03 | Data and metric interfaces; participant separation tests | November 1-2 |
| T04-T06 | Environment checks, initial paper structure, exploratory analysis | November 2 |
| T07 | Optional Logistic Regression baseline | November 4 |
| T08-T09 | SVM and Random Forest validation searches | November 5 |
| T10 | Introduction, related work, dataset, methodology draft | November 6 |
| T11 | Freeze model settings | November 5 |
| T12 | Final official test evaluation | November 6-7 |
| T13-T15 | Error analysis, feature analysis, model comparison | November 8 |
| T16-T17 | Figures and remaining paper sections | November 9-10 |
| T18-T19 | Consolidation and frozen rerun in a clean environment | November 11 |
| T20-T21 | Final checks and confirmed submission | November 12-13 |
| T22 | Optional KNN after required work | November 7-9 |

These are planning targets, not completed tasks. Schedule a short daily check-in to record completed work, next steps, and blockers.

## Planned implementation

Use Python and scikit-learn, with NumPy/pandas for data and Matplotlib for figures. Agree on one Python version and record tested dependency versions. Avoid claiming installation commands work until verified.

Start with data loading and participant validation. Then implement the two models using shared folds and metric definitions. Add experiment commands and result plots once these pieces work. Introduce `src/`, `tests/`, configurations, notebooks, and results as actual content becomes available.

## Reproducibility and results

Save per-model parameters, validation means and standard deviations, final test metrics, confusion matrices, search and fit times, prediction times, seed, dependency versions, and hardware details. Associate outputs with the code revision. Keep validation and test results clearly separate. Never enter invented results or manually change numbers in generated tables.

The paper should cover introduction, related work, dataset and methodology, models, results, discussion and limitations, conclusion, and references. Write methods before experiments and results after final evaluation.

## Risks and fallback

If time is short, remove KNN, then the optional baseline, then reduce the validation search budget before consulting the test set. Keep time for writing. If leakage is found, correct it and rerun affected experiments, documenting the issue. Keep raw data unchanged and out of version control.

## Completion checks

- All members can access the private repository.
- Data and participant separation checks pass.
- Both required models use comparable validation and evaluation.
- Results and figures come from recorded experiments.
- A clean environment can reproduce the frozen procedure.
- README commands match implemented and tested behavior.
- All research questions are answered with evidence and limitations.
- Paper requirements and submission are confirmed with the instructor.
