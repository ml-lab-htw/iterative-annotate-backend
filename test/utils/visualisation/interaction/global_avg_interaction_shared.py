import json
import math
import os
from dotenv import load_dotenv
from numpy.ma.core import left_shift

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

def calculate_global_avg_times(manual_data, reviewed_data):
    interaction_types = ['boxCreatedCnt', 'boxRemovedCnt', 'boxMovedCnt', 'labelUpdatedCnt', 'navigateImgCnt']

    total_duration = 0
    total_interactions = {it: 0 for it in interaction_types}

    for data in [manual_data, reviewed_data]:
        for bundle in data:
            total_duration += bundle['labeling']['duration'] / 1000  # Convert to seconds
            for it in interaction_types:
                total_interactions[it] += bundle['labeling'][it]

    total_interaction_count = sum(total_interactions.values())

    if total_interaction_count > 0:
        global_avg_time = total_duration / total_interaction_count
        avg_times = {it: global_avg_time for it in interaction_types}
    else:
        avg_times = {it: 0 for it in interaction_types}

    return avg_times, total_interactions, total_duration

def calculate_times_per_bundle(data, global_avg_times):
    interaction_types = ['boxCreatedCnt', 'boxRemovedCnt', 'boxMovedCnt', 'labelUpdatedCnt', 'navigateImgCnt']

    times_per_bundle = []
    for bundle in data:
        bundle_times = {it: global_avg_times[it] * bundle['labeling'][it] for it in interaction_types}
        times_per_bundle.append(bundle_times)

    return times_per_bundle

def main(run_name):
    # Load all data from json
    manual_data, reviewed_data = load_json_files(run_name)

    bundles = list(range(1, len(manual_data) + 1))

    # Calculate global average times
    global_avg_times, total_interactions, total_duration = calculate_global_avg_times(manual_data, reviewed_data)

    # Calculate times for each bundle using the global average times
    manual_times_per_bundle = calculate_times_per_bundle(manual_data, global_avg_times)
    semi_auto_times_per_bundle = calculate_times_per_bundle(reviewed_data, global_avg_times)

    # Prepare data for stacked bar chart
    interaction_types = ['boxCreatedCnt', 'boxRemovedCnt', 'boxMovedCnt', 'labelUpdatedCnt', 'navigateImgCnt']
    interaction_labels = ['Annotation created', 'Annotation removed', 'Annotation moved', 'Label updated',
                          'Image navigated']
    colors = ['#F6DCAC', '#FEAE6F', '#E85917', '#028391', '#01204E']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 14), sharex=True)  # Increased figure height

    # Manual process
    bottom = np.zeros(len(bundles))
    for i, it in enumerate(interaction_types):
        values = [times[it] for times in manual_times_per_bundle]
        ax1.bar(bundles, values, bottom=bottom, label=interaction_labels[i], color=colors[i], edgecolor='black',
                linewidth=0.5)
        bottom += values

    ax1.set_title('Manual process (using global average times)', fontsize=26, loc='left')
    ax1.set_xlabel('Bundle id', fontsize=22)
    ax1.set_ylabel('Time (seconds)', fontsize=22)

    # Semi-automated process
    bottom = np.zeros(len(bundles))
    for i, it in enumerate(interaction_types):
        values = [times[it] for times in semi_auto_times_per_bundle]
        ax2.bar(bundles, values, bottom=bottom, label=interaction_labels[i], color=colors[i], edgecolor='black',
                linewidth=0.5)
        bottom += values

    ax2.set_title('Semi-automated process (using global average times)', fontsize=26, loc='left')
    ax2.set_xlabel('Bundle id', fontsize=22)
    ax2.set_ylabel('Time (seconds)', fontsize=22)

    # Calculate the maximum y-value across both charts
    manual_max = max(sum(bundle.values()) for bundle in manual_times_per_bundle)
    semi_auto_max = max(sum(bundle.values()) for bundle in semi_auto_times_per_bundle)
    overall_max = max(manual_max, semi_auto_max)

    # Set x and y limits for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlim(.5, len(bundles) + 0.5)
        ax.set_ylim(0, math.ceil(overall_max) + 10)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.xticks(bundles)

    # Create a shared legend with larger patches and black edges
    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06),
                        fontsize=19, frameon=True, handleheight=2, handlelength=2)

    # Add black edges to legend patches
    for handle in legend.legend_handles:
        handle.set_edgecolor('black')
        handle.set_linewidth(1)

    # Adjust the layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, hspace=0.15)

    save_path = os.path.join(base_path, script_path, f'pdf_export/new_bundle_global_avg_{run_name}.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    # Print global average times and total interactions for reference
    print("Global average time per interaction:", global_avg_times[interaction_types[0]])
    print("\nTotal interactions across both processes:")
    for it, count in total_interactions.items():
        print(f"{it}: {count}")

    print(f"\nTotal duration across both processes: {total_duration:.2f} seconds")

    # Verification
    total_calculated_time = sum(sum(bundle.values()) for bundle in manual_times_per_bundle) + \
                            sum(sum(bundle.values()) for bundle in semi_auto_times_per_bundle)
    print(f"\nTotal calculated time: {total_calculated_time:.2f} seconds")
    print(f"Difference: {abs(total_duration - total_calculated_time):.2f} seconds")

if __name__ == "__main__":
    run_id = "n_10"
    main(run_id)