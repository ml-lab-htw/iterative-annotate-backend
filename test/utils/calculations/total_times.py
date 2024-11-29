import json
from pathlib import Path
import numpy as np

base_path = "../../dead-tree/sub-dataset"


def load_json_files(run_name):
    paths = {
        "manual": Path(f"{base_path}/{run_name}/_eval/manual.json"),
        "review": Path(f"{base_path}/{run_name}/_eval/reviewed.json")
    }

    with open(paths["manual"], "r") as f:
        manual_data = json.load(f)

    with open(paths["review"], "r") as f:
        reviewed_data = json.load(f)

    return manual_data, reviewed_data


def calculate_global_avg_time(manual_data, reviewed_data):
    interaction_types = ['boxCreatedCnt', 'boxRemovedCnt', 'boxMovedCnt', 'labelUpdatedCnt', 'navigateImgCnt']

    total_duration = 0
    total_interactions = 0

    for data in [manual_data, reviewed_data]:
        for bundle in data:
            total_duration += bundle['labeling']['duration'] / 1000  # Convert to seconds
            total_interactions += sum(bundle['labeling'][it] for it in interaction_types)

    global_avg_time = total_duration / total_interactions if total_interactions > 0 else 0

    return global_avg_time


def calculate_total_times(data, global_avg_time):
    interaction_types = ['boxCreatedCnt', 'boxRemovedCnt', 'boxMovedCnt', 'labelUpdatedCnt', 'navigateImgCnt']

    total_plain_duration = 0
    total_global_avg_duration = 0

    for bundle in data:
        total_plain_duration += bundle['labeling']['duration'] / 1000  # Convert to seconds

        total_interactions = sum(bundle['labeling'][it] for it in interaction_types)
        total_global_avg_duration += global_avg_time * total_interactions

    return total_plain_duration, total_global_avg_duration


def main(run_name):
    manual_data, reviewed_data = load_json_files(run_name)
    global_avg_time = calculate_global_avg_time(manual_data, reviewed_data)

    manual_plain_total, manual_global_total = calculate_total_times(manual_data, global_avg_time)
    semi_auto_plain_total, semi_auto_global_total = calculate_total_times(reviewed_data, global_avg_time)

    print(f"Analysis of {len(manual_data)}-bundle run, with {run_name} images per bundle:")
    print("\nTotal Time for Complete Test Run:")
    print("Manual Labeling:")
    print(f"  Plain Duration: {manual_plain_total:.2f} seconds")
    print(f"  Global Average Duration: {manual_global_total:.2f} seconds")
    print("Semi-Automatic Labeling:")
    print(f"  Plain Duration: {semi_auto_plain_total:.2f} seconds")
    print(f"  Global Average Duration: {semi_auto_global_total:.2f} seconds")

    print("\nTime Savings:")
    plain_savings = manual_plain_total - semi_auto_plain_total
    global_savings = manual_global_total - semi_auto_global_total
    print(f"  Plain Duration Savings: {plain_savings:.2f} seconds")
    print(f"  Global Average Duration Savings: {global_savings:.2f} seconds")

    print("\nEfficiency Improvement:")
    plain_improvement = (manual_plain_total - semi_auto_plain_total) / manual_plain_total * 100
    global_improvement = (manual_global_total - semi_auto_global_total) / manual_global_total * 100
    print(f"  Plain Duration Improvement: {plain_improvement:.2f}%")
    print(f"  Global Average Duration Improvement: {global_improvement:.2f}%")


if __name__ == "__main__":
    run_id = "n_10"
    main(run_id)