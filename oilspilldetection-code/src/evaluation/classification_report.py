import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def generate_classification_report(y_true, y_pred, output_dir="."):
    """
    Generates classification report & saves metrics for Oil Spill Detection
    """

    target_names = ["Normal", "Abnormal"]

    # Generate text report
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=2
    )

    print("\nClassification Report:\n")
    print(report)

    # Save text report
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names
    )

    cm_df.to_csv(f"{output_dir}/offline_confusion_matrix.csv")

    # Extract metrics into table
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True
    )

    metrics_df = pd.DataFrame(report_dict).transpose()
    metrics_df.to_csv(f"{output_dir}/offline_model_metrics.csv")

    print("\nSaved files:")
    print(" - classification_report.txt")
    print(" - offline_confusion_matrix.csv")
    print(" - offline_model_metrics.csv")

    return report, cm_df, metrics_df
