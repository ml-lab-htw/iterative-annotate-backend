import json
import math
import os
from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import numpy as np


base_path = os.getenv('BASE_DIR')
data_path = "test/crop-detection/sub-dataset"
script_path = "test/utils/visualisation/efficiency"
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

    return [
        bundle["labeling"]["duration"] / 1000
        for bundle in manual_data
    ],  [
        bundle["labeling"]["duration"] / 1000
        for bundle in reviewed_data
    ]

def main(run_name):
    # Data
    manual, semi_auto = load_json_files(run_name)
    bundles = list(range(1, len(manual)+1))
    print(manual)
    print(semi_auto)

    # Calculate absolute difference
    difference = np.abs(np.array(manual) - np.array(semi_auto))

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the shaded area for absolute difference
    plt.fill_between(bundles, 0, difference, alpha=0.5, label='Absolute difference', color='#F6DCAC')

    # Plot the lines for manual and semi-automated
    plt.plot(bundles, manual, label='Manual', marker='o', color='#E85917')
    plt.plot(bundles, semi_auto, label='Semi-automated', marker='o', color='#028391')

    # Customize the plot
    #plt.title('Comparison of labeling Duration: Manual vs Semi-Automated')
    plt.xlabel('Bundle id', fontsize=20)
    plt.ylabel('Labeling duration (seconds)', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.8)

    # Set axis limits
    plt.xlim(1, len(bundles)-1)
    plt.ylim(0, math.floor(max([max(manual), max(semi_auto)])) + 25)

    # Set plot padding
    plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.08)

    # Ensure integer ticks on x-axis
    plt.xticks(bundles)

    # Save the plot as a PDF
    save_path = os.path.join(base_path, script_path, f'pdf_export/duration_chart_{run_name}.pdf')
    plt.savefig(save_path, bbox_inches='tight')

    # Display the plot (optional, comment out if not needed)
    plt.show()

if __name__ == "__main__":
    run_id = "n_10"
    main(run_id)