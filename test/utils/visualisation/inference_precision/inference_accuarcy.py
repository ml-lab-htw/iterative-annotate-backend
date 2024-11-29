import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
base_path = os.getenv('BASE_DIR')

# Paths setup
data_path = "test/dead-tree/sub-dataset"
script_path = "test/utils/visualisation/inference_precision"
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

    for idx, eval_data in enumerate(data):
        tp = eval_data['true_positives']
        fp = eval_data['false_positives']
        fn = eval_data['false_negatives']
        mean_iou = eval_data['mean_iou']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        steps.append(idx + 1)
        precisions.append(precision)
        recalls.append(recall)
        mean_ious.append(mean_iou)

    return steps, precisions, recalls, mean_ious


def plot_metrics(steps, precisions, recalls, mean_ious, run_id, data_type):
    plt.figure(figsize=(12, 6))
    plt.tight_layout()

    plt.plot(steps, precisions, label='Precision', marker='o', color='#E85917')
    plt.plot(steps, recalls, label='Recall', marker='s', color='#01204E')
    plt.plot(steps, mean_ious, label='Mean IoU', marker='^', color='#028391')

    plt.title(f'{data_type.capitalize()}: Precision, recall and mean IoU over bundle step')
    plt.xlabel('Bundle Id')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Set axis limits dynamically
    plt.xlim(min(steps), max(steps))
    plt.ylim(0, 1.05)

    # Set x-axis ticks dynamically
    plt.xticks(range(min(steps), max(steps) + 1))

    # Save the plot as a PDF
    output_path = os.path.join(base_path, script_path, f"pdf_2nd_export/{data_type}_accuracy_{run_id}.pdf")
    plt.savefig(output_path, bbox_inches='tight', format='pdf')
    plt.show()


def main(run_id):
    data = load_json_files(run_id)

    for data_type in ["manual", "inference", "review"]:
        steps, precisions, recalls, mean_ious = compute_metrics(data[data_type])
        plot_metrics(steps, precisions, recalls, mean_ious, run_id, data_type)

if __name__ == "__main__":
    run_id = "n_10"
    main(run_id)
