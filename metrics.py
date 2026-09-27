import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_recall_fscore_support
)

# 1. Define DR Severity Grade Labels
GRADE_LABELS = ["Grade 0 (No DR)", "Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)", "Grade 4 (Proliferative)"]

def calculate_referable_dr_metrics(y_true, y_pred):
    """
    Evaluates Referable DR (Grade 2+) against PS Targets:
    Target: Sensitivity > 90%, Specificity > 85%
    """
    # Binary conversion: Non-Referable (0, 1) vs Referable (2, 3, 4)
    y_true_binary = (np.array(y_true) >= 2).astype(int)
    y_pred_binary = (np.array(y_pred) >= 2).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        "sensitivity": sensitivity * 100,
        "specificity": specificity * 100,
        "precision": precision * 100,
        "f1_score": f1 * 100,
        "pass_sensitivity": sensitivity >= 0.90,
        "pass_specificity": specificity >= 0.85
    }

def generate_validation_plots(y_true, y_pred, y_probs=None):
    """
    Generates Confusion Matrix and ROC Curves.
    """
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cax = ax_cm.matshow(cm, cmap=plt.cm.Blues)
    fig_cm.colorbar(cax)

    for i in range(5):
        for j in range(5):
            ax_cm.text(j, i, str(cm[i, j]), va='center', ha='center', color='red' if cm[i, j] > 0 else 'black')

    ax_cm.set_xticks(range(5))
    ax_cm.set_yticks(range(5))
    ax_cm.set_xticklabels([f"G{i}" for i in range(5)])
    ax_cm.set_yticklabels([f"G{i}" for i in range(5)])
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title("5x5 DR Grade Confusion Matrix")
    plt.tight_layout()

    return fig_cm

if __name__ == "__main__":
    # Synthetic Validation Test Run
    np.random.seed(42)
    y_true_demo = np.random.choice([0, 1, 2, 3, 4], size=200, p=[0.4, 0.25, 0.2, 0.1, 0.05])
    
    # Simulate high model accuracy (~88%)
    y_pred_demo = y_true_demo.copy()
    noise_idx = np.random.choice(200, size=24, replace=False)
    y_pred_demo[noise_idx] = np.random.choice([0, 1, 2, 3, 4], size=24)

    metrics = calculate_referable_dr_metrics(y_true_demo, y_pred_demo)
    print("=== Validation Results ===")
    print(f"Referable Sensitivity: {metrics['sensitivity']:.2f}% (Target > 90%: {metrics['pass_sensitivity']})")
    print(f"Referable Specificity: {metrics['specificity']:.2f}% (Target > 85%: {metrics['pass_specificity']})")