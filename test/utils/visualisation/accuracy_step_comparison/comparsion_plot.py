import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
base_path = os.getenv('BASE_DIR')

# Paths setup
data_path = "test/dead-tree/sub-dataset"
script_path = "test/utils/visualisation/accuracy_step_comparison"
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
            data[key] = [item.get("eval") for item in json.load(f)]

    return data

def compute_metrics(data):
    steps = []
    precisions = []
    recalls = []
    mean_ious = []
    f1_scores = []

    for idx, eval_data in enumerate(data):
        tp = eval_data['true_positives']
        fp = eval_data['false_positives']
        fn = eval_data['false_negatives']
        mean_iou = eval_data['mean_iou']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        steps.append(idx + 1)
        precisions.append(precision)
        recalls.append(recall)
        mean_ious.append(mean_iou)
        f1_scores.append(f1_score)

    return steps, precisions, recalls, mean_ious, f1_scores

def plot_metric(steps, metric_values, metric_name, run_id, data_types, colors):
    plt.figure(figsize=(12, 6))
    plt.tight_layout()

    for data_type, values in metric_values.items():
        plt.plot(steps, values, label=data_type.capitalize(), marker='o', color=colors[data_type])

    plt.title(f'{metric_name.capitalize()} over bundle steps')
    plt.xlabel('Bundle Id')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)

    # Set fixed axis limits
    plt.xlim(min(steps), max(steps))
    plt.ylim(0, 1.05)

    # Set x-axis ticks dynamically
    plt.xticks(range(min(steps), max(steps) + 1))

    # Save the plot as a PDF
    output_path = os.path.join(base_path, script_path, f"pdf_2nd_export/{metric_name}_{run_id}.pdf")
    plt.savefig(output_path, bbox_inches='tight', format='pdf')
    plt.show()

def main(run_id):
    data = load_json_files(run_id)

    steps = None
    metric_values = {
        'manual': {},
        'inference': {},
        'review': {}
    }

    for data_type in ["manual", "inference", "review"]:
        steps, precisions, recalls, mean_ious, f1_scores = compute_metrics(data[data_type])
        metric_values[data_type] = {
            'precision': precisions,
            'recall': recalls,
            'iou': mean_ious,
            'f1': f1_scores
        }

    colors = {
        'manual': '#E85917',
        'inference': '#01204E',
        'review': '#028391'
    }

    # Plot each metric across all data types
    for metric_name in ['precision', 'recall', 'iou', 'f1']:
        plot_metric(steps, {dt: metric_values[dt][metric_name] for dt in metric_values}, metric_name, run_id, metric_values, colors)


if __name__ == "__main__":
    run_id = "n_10"
    main(run_id)
