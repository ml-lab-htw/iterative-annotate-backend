import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

base_path = "../../dead-tree/sub-dataset"

def load_json_files(run_name):
    paths = {
        "manual": Path(f"{base_path}/{run_name}/_eval/manual.json"),
        "review": Path(f"{base_path}/{run_name}/_eval/reviewed.json")
    }

    with open(paths.get("manual"), "r") as f:
        manual_data = json.load(f)

    with open(paths.get("review"), "r") as f:
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

def calculate_global_avg_durations(data, global_avg_time):
    interaction_types = ['boxCreatedCnt', 'boxRemovedCnt', 'boxMovedCnt', 'labelUpdatedCnt', 'navigateImgCnt']

    global_avg_durations = []
    for bundle in data:
        total_interactions = sum(bundle['labeling'][it] for it in interaction_types)
        duration = global_avg_time * total_interactions
        global_avg_durations.append(duration)

    return global_avg_durations

def calculate_relative_improvement(manual_avg, semi_auto_avg):
    relative_diff = (manual_avg - semi_auto_avg) / manual_avg * 100 if manual_avg != 0 else 0
    return relative_diff

def calculate_efficiency_gain(start, end):
    return (start - end) / start * 100 if start != 0 else 0

def main(run_name):
    manual_data, reviewed_data = load_json_files(run_name)
    global_avg_time = calculate_global_avg_time(manual_data, reviewed_data)

    # Actual durations
    manual_actual_durations = [bundle['labeling']['duration'] / 1000 for bundle in manual_data]
    semi_auto_actual_durations = [bundle['labeling']['duration'] / 1000 for bundle in reviewed_data]

    # Durations calculated using global average time
    manual_global_avg_durations = calculate_global_avg_durations(manual_data, global_avg_time)
    semi_auto_global_avg_durations = calculate_global_avg_durations(reviewed_data, global_avg_time)

    n_epochs_to_compare = 3  # You can adjust this number

    # Calculate averages for first and last epochs
    manual_actual_first = np.mean(manual_actual_durations[:n_epochs_to_compare])
    manual_actual_last = np.mean(manual_actual_durations[-n_epochs_to_compare:])
    semi_auto_actual_first = np.mean(semi_auto_actual_durations[:n_epochs_to_compare])
    semi_auto_actual_last = np.mean(semi_auto_actual_durations[-n_epochs_to_compare:])

    manual_global_first = np.mean(manual_global_avg_durations[:n_epochs_to_compare])
    manual_global_last = np.mean(manual_global_avg_durations[-n_epochs_to_compare:])
    semi_auto_global_first = np.mean(semi_auto_global_avg_durations[:n_epochs_to_compare])
    semi_auto_global_last = np.mean(semi_auto_global_avg_durations[-n_epochs_to_compare:])

    # Calculate relative improvements
    actual_improvement_first = calculate_relative_improvement(manual_actual_first, semi_auto_actual_first)
    actual_improvement_last = calculate_relative_improvement(manual_actual_last, semi_auto_actual_last)
    global_improvement_first = calculate_relative_improvement(manual_global_first, semi_auto_global_first)
    global_improvement_last = calculate_relative_improvement(manual_global_last, semi_auto_global_last)

    # Calculate efficiency gains
    manual_actual_gain = calculate_efficiency_gain(manual_actual_first, manual_actual_last)
    semi_auto_actual_gain = calculate_efficiency_gain(semi_auto_actual_first, semi_auto_actual_last)
    manual_global_gain = calculate_efficiency_gain(manual_global_first, manual_global_last)
    semi_auto_global_gain = calculate_efficiency_gain(semi_auto_global_first, semi_auto_global_last)

    # Calculate differences in improvement
    actual_diff = actual_improvement_last - actual_improvement_first
    global_avg_diff = global_improvement_last - global_improvement_first

    print(f"Analysis of {len(manual_data)}-bundle run, with {run_name} images per bundle:")
    print("\nActual Recorded Durations:")
    print(f"Manual Start: {manual_actual_first:.2f} seconds")
    print(f"Manual Finish: {manual_actual_last:.2f} seconds")
    print(f"Semi-auto Start: {semi_auto_actual_first:.2f} seconds")
    print(f"Semi-auto Finish: {semi_auto_actual_last:.2f} seconds")
    print(f"Initial Improvement: {actual_improvement_first:.2f}%")
    print(f"Final Improvement: {actual_improvement_last:.2f}%")
    print(f"Change in Improvement: {actual_diff:.2f}%")
    print(f"Manual Efficiency Gain: {manual_actual_gain:.2f}%")
    print(f"Semi-auto Efficiency Gain: {semi_auto_actual_gain:.2f}%")

    print("\nDurations Calculated with Global Average Time:")
    print(f"Manual Start: {manual_global_first:.2f} seconds")
    print(f"Manual Finish: {manual_global_last:.2f} seconds")
    print(f"Semi-auto Start: {semi_auto_global_first:.2f} seconds")
    print(f"Semi-auto Finish: {semi_auto_global_last:.2f} seconds")
    print(f"Initial Improvement: {global_improvement_first:.2f}%")
    print(f"Final Improvement: {global_improvement_last:.2f}%")
    print(f"Change in Improvement: {global_avg_diff:.2f}%")
    print(f"Manual Efficiency Gain: {manual_global_gain:.2f}%")
    print(f"Semi-auto Efficiency Gain: {semi_auto_global_gain:.2f}%")

    print(f"\nDiscrepancy: {abs(actual_diff - global_avg_diff):.2f}%")

if __name__ == "__main__":
    run_id = "n_10"
    main(run_id)