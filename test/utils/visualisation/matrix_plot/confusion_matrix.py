import json
import os
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base_path = os.getenv('BASE_DIR')
data_path = "test/crop-detection/sub-dataset"
script_path = "test/utils/visualisation/matrix_plot"
data_dir = os.path.join(base_path, data_path)


def load_json_files(run_name):
    paths = {
        "manual": os.path.join(data_dir, f"{run_name}/_eval/manual.json"),
        "review": os.path.join(data_dir, f"{run_name}/_eval/reviewed.json"),
        "inference": os.path.join(data_dir, f"{run_name}/_eval/inference.json"),
    }

    data = {}
    for key, path in paths.items():
        with open(path, "r") as f:
            data[key] = [{"bundle_id": item["bundle_id"], "eval": item["eval"]} for item in json.load(f)]

    return data


def create_confusion_matrix(eval_data):
    TP = float(eval_data["true_positives"])
    FP = float(eval_data["false_positives"])
    FN = float(eval_data["false_negatives"])

    return np.array([[TP, FN], [FP, 0.0], [0.0, 0.0]])


def plot_confusion_matrix(cm, title, output_path=None):
    plt.figure(figsize=(8, 6))
    plt.tight_layout()

    # Custom format function to handle both int and float
    def format_val(val):
        if val == 0:
            return '0'
        elif abs(val) < 0.01:
            return f'{val:.2e}'
        elif val.is_integer():
            return f'{int(val)}'
        else:
            return f'{val:.2f}'

    # Create the heatmap
    ax = sns.heatmap(cm, annot=False, cmap='Blues',
                     xticklabels=['Positive', 'Negative'],
                     yticklabels=['True positive', 'False positive', 'False negative'],
                     cbar=False)

    # Get the color normalization
    norm = plt.Normalize(cm.min(), cm.max())

    # Add text annotations with dynamic color
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = plt.cm.Blues(norm(cm[i, j]))

            # Calculate the luminance of the background color
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

            # Choose white text for dark backgrounds, black text for light backgrounds
            text_color = "white" if luminance < 0.5 else "black"

            ax.text(j + 0.5, i + 0.5, format_val(cm[i, j]),
                    ha="center", va="center", color=text_color,
                    fontweight='normal')

    plt.title(title)
    plt.ylabel('Predicted')
    plt.xlabel('Actual (ground truth)')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    plt.close()

def calculate_metrics(cm):
    TP, FN = cm[0]
    FP, _ = cm[1]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }


def calculate_mean_eval(bundles):
    mean_eval = {
        "true_positives": np.mean([b["eval"]["true_positives"] for b in bundles]),
        "false_positives": np.mean([b["eval"]["false_positives"] for b in bundles]),
        "false_negatives": np.mean([b["eval"]["false_negatives"] for b in bundles])
    }
    return mean_eval


def main(run_id):
    data = load_json_files(run_id)

    for data_type in ["manual", "inference", "review"]:
        bundles = data[data_type]

        if data_type == "inference":
            bundle_indices = [0, len(bundles) // 2, -1]  # First and last bundle indices
            bundle_names = ["First", "Middle", "Last"]

            for idx, bundle_idx in enumerate(bundle_indices):
                bundle = bundles[bundle_idx]
                cm = create_confusion_matrix(bundle["eval"])

                title = f"{data_type.capitalize()} vs ground truth - {bundle_names[idx]} Bundle (ID: {bundle['bundle_id']})"
                save_path = os.path.join(base_path, script_path,
                                         f'pdf_export/confusion_matrix_{data_type}_{bundle_names[idx].lower()}_{run_id}.pdf')
                plot_confusion_matrix(cm, title, save_path)

                print(f"\n{title}")
                print(cm)

                metrics = calculate_metrics(cm)
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
        else:
            mean_eval = calculate_mean_eval(bundles)
            cm = create_confusion_matrix(mean_eval)

            title = f"{data_type.capitalize()} vs ground truth - Mean Values"
            save_path = os.path.join(base_path, script_path,
                                     f'pdf_export/confusion_matrix_{data_type}_mean_{run_id}.pdf')
            plot_confusion_matrix(cm, title, save_path)

            print(f"\n{title}")
            print(cm)

            metrics = calculate_metrics(cm)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    run_id = "n_25"
    main(run_id)