import json
import math
import os
from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import numpy as np

base_path = os.getenv('BASE_DIR')
data_path = "test/crop-detection/sub-dataset"
script_path = "test/utils/visualisation/interaction"
data_dir = os.path.join(base_path, data_path)

def load_json_files(run_name):
    paths = {
        "manual": os.path.join(data_dir, f"{run_name}/_eval/manual.json"),
        "review": os.path.join(data_dir, f"{run_name}/_eval/reviewed.json"),
    }

    with open(paths.get("manual"), "r") as f:
        manual_data = json.load(f)

    with open(paths.get("review"), "r") as f:
        reviewed_data = json.load(f)

    return manual_data, reviewed_data


def calculate_avg_time_per_interaction(data):
    interaction_types = ['boxCreatedCnt', 'boxRemovedCnt', 'boxMovedCnt', 'labelUpdatedCnt', 'navigateImgCnt']

    avg_times = []
    for bundle in data:
        total_time = bundle['labeling']['duration'] / 1000  # Convert to seconds
        total_interactions = sum(bundle['labeling'][it] for it in interaction_types)

        if total_interactions > 0:
            avg_time = total_time / total_interactions
            avg_times.append({it: avg_time * bundle['labeling'][it] for it in interaction_types})
        else:
            avg_times.append({it: 0 for it in interaction_types})

    return avg_times


def main(run_name):
    # Data
    # Load all data from json
    manual_data, reviewed_data = load_json_files(run_name)

    bundles = list(range(1, len(manual_data) + 1))

    # Calculate average time per interaction
    manual_avg_times = calculate_avg_time_per_interaction(manual_data)
    semi_auto_avg_times = calculate_avg_time_per_interaction(reviewed_data)

    # Prepare data for stacked bar chart
    interaction_types = ['boxCreatedCnt', 'boxRemovedCnt', 'boxMovedCnt', 'labelUpdatedCnt', 'navigateImgCnt']
    interaction_labels = ['Annotation created', 'Annotation removed', 'Annotation moved', 'Label updated', 'Image navigated']
    colors = ['#F6DCAC', '#FEAE6F', '#E85917', '#028391', '#01204E']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Manual process
    bottom = np.zeros(len(bundles))
    for i, it in enumerate(interaction_types):
        values = [times[it] for times in manual_avg_times]
        ax1.bar(bundles, values, bottom=bottom, label=interaction_labels[i], color=colors[i])
        bottom += values

    ax1.set_title('Manual process')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Semi-automated process
    bottom = np.zeros(len(bundles))
    for i, it in enumerate(interaction_types):
        values = [times[it] for times in semi_auto_avg_times]
        ax2.bar(bundles, values, bottom=bottom, label=interaction_labels[i], color=colors[i])
        bottom += values

    ax2.set_title('Semi-automated process')
    ax2.set_xlabel('Bundle id')
    ax2.set_ylabel('Time (seconds)')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Calculate the maximum y-value across both charts
    manual_max = max(sum(bundle.values()) for bundle in manual_avg_times)
    semi_auto_max = max(sum(bundle.values()) for bundle in semi_auto_avg_times)
    overall_max = max(manual_max, semi_auto_max)

    # Set x and y limits for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlim(.5, len(bundles)+0.5)
        ax.set_ylim(0, math.floor(overall_max) + 10)

    plt.xticks(bundles)
    plt.tight_layout()
    save_path = os.path.join(base_path, script_path, f'pdf_export/bundle_{run_name}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_id = "n_25"
    main(run_id)