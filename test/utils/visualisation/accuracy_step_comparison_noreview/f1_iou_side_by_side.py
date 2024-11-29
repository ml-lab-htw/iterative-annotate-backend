import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
base_path = os.getenv('BASE_DIR')

# Paths setup
data_path = "test/crop-detection/sub-dataset"
script_path = "test/utils/visualisation/accuracy_step_comparison_noreview"
data_dir = os.path.join(base_path, data_path)

def load_json_files(run_name):
    paths = {
        "manual": os.path.join(data_dir, f"{run_name}/_eval/manual.json"),
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

def plot_metrics_combined(steps, metric_values, run_id, colors, custom_labels):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(wspace=0.1)  # Reduced spacing between subplots

    # Metrics to plot
    metrics = ['f1', 'iou']
    metric_titles = {'f1': 'F1 Score', 'iou': 'Mean IoU'}

    for i, metric_name in enumerate(metrics):
        ax = axs[i]
        for data_type in ['manual', 'inference']:
            ax.plot(steps, metric_values[data_type][metric_name], label=custom_labels[data_type],
                    marker='o', color=colors[data_type])
        ax.set_title(metric_titles[metric_name])
        ax.set_xlabel('Bundle Step')
        ax.grid(True)
        ax.set_xlim(min(steps), max(steps))
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(min(steps), max(steps) + 1))
        if i == 0:
            ax.set_ylabel('Score')

    # Adjust legend to have labels stacked vertically
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.05))

    # Save the plot as a PDF
    output_path = os.path.join(base_path, script_path, f"pdf_export/f1_iou_combined_{run_id}.pdf")
    plt.savefig(output_path, bbox_inches='tight', format='pdf')
    plt.show()

def main(run_id):
    data = load_json_files(run_id)

    metric_values = {
        'manual': {},
        'inference': {}
    }

    # Initialize steps to be used for plotting
    steps_list = []

    for data_type in ["manual", "inference"]:
        steps, precisions, recalls, mean_ious, f1_scores = compute_metrics(data[data_type])

        # Determine the number of steps available
        num_steps = len(steps)
        max_steps = min(num_steps, 10)  # Ensure we don't exceed 10 steps

        # Limit the data to the first 10 steps or available steps
        steps = steps[:max_steps]
        precisions = precisions[:max_steps]
        recalls = recalls[:max_steps]
        mean_ious = mean_ious[:max_steps]
        f1_scores = f1_scores[:max_steps]

        # Store steps for plotting (assuming both data types have the same steps)
        if not steps_list or len(steps) < len(steps_list):
            steps_list = steps

        metric_values[data_type] = {
            'precision': precisions,
            'recall': recalls,
            'iou': mean_ious,
            'f1': f1_scores
        }

    colors = {
        'manual': '#E85917',
        'inference': '#01204E',
    }

    # Custom labels for the legend
    custom_labels = {
        'manual': 'Manual',
        'inference': 'Semi-automatic'
    }

    # Plot the combined F1 Score and IoU metrics
    plot_metrics_combined(steps_list, metric_values, run_id, colors, custom_labels)

if __name__ == "__main__":
    run_id = "n_25"
    main(run_id)
