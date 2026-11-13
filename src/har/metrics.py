from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ACTIVITY_LABELS = (1, 2, 3, 4, 5, 6)

def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ =  precision_recall_fscore_support(
        y_true, y_pred,
        labels = ACTIVITY_LABELS,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ACTIVITY_LABELS),
    }