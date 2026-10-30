# Motion ID research proposal

## Research problem

Smartphone motion sensors record acceleration and rotation without directly identifying the activity being performed. This study investigates whether machine learning can recognize six everyday activities from those measurements for people who did not participate in training.

## Research questions

1. How well can the six activities be recognized for unseen participants?
2. Which model, SVM or Random Forest, offers the best recognition performance and computational cost?
3. Which activities are most often confused, and which sensor features help distinguish them?

## Dataset

Use the original UCI Human Activity Recognition Using Smartphones dataset: 10,299 windows, 561 calculated features, 30 participants, and six activities. Retain the official split of 7,352 training windows and 2,947 test windows, collected from different participants. See [dataset details and references](dataset.md).

## Candidate models and rationale

- **SVM:** a candidate for separating activities using a large set of numerical features. Standardize features inside the training pipeline and tune a limited set of settings using participant-based validation.
- **Random Forest:** a complementary tree-based model that can represent nonlinear relationships and provide feature importance estimates. Feature importance will support interpretation, without being treated as proof of causation.
- **Logistic Regression, optional:** a simpler reference model to assess whether the main models justify their additional complexity.

These are hypotheses to evaluate. No model is assumed to outperform another before experiments.

## Evaluation

Use the same participant-based validation folds for all models. Keep the official test set reserved until model settings are frozen. Report accuracy, macro precision, macro recall, macro F1, confusion matrices, training time, and prediction time. Define the main selection metric and search budget before running the comparison.

## Expected contribution

A reproducible comparison, an analysis of classification errors, and a discussion of the relationship between recognition performance and processing cost. Expected deliverables are research code, machine-generated tables and figures, and a paper. No experimental findings are available yet.

## First-cycle submission

The supplied course brief requests the research problem, research questions, dataset, and possible models with justification, delivered as a PDF. This Markdown document is the editable proposal source; the Portuguese [submission PDF](proposal.pdf) is available alongside it. The final paper format, deadline, and page limit require confirmation with the instructor.
