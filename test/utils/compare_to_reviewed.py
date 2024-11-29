import json
from pathlib import Path
from box_calculations import extract_bundle_number, evaluate_bundle

base_path = "../dead-tree/sub-dataset"


def main(name):

    ground_truth_path = Path(f"{base_path}/{name}/subset_data.json")
    with open(ground_truth_path, "r") as f:
        ground_truth_data = json.load(f)

    bundles = {}
    for image_data in ground_truth_data:
        bundle_id = Path(image_data["image"]).parent.name
        if bundle_id not in bundles:
            bundles[bundle_id] = []
        bundles[bundle_id].append(image_data)

    data_group = []
    for bundle_id, ground_truth in bundles.items():
        reviewed_path = Path(f"{base_path}/{name}/{bundle_id}/_review_labeling.json")
        with open(reviewed_path, "r") as f:
            json_data = json.load(f)

        reviewed_data = json_data["images"]
        bundle_nbr = extract_bundle_number(bundle_id)

        metrics = evaluate_bundle(ground_truth, reviewed_data)

        print(f"{bundle_nbr}. - {metrics['average_precision']}")
        data_group.append({
            "bundle_id": bundle_nbr,
            "labeling": {
                "duration": json_data["duration"],
                **json_data["info"]
            },
            "training": json_data["trainings_info"],
            "eval": metrics
        })
    return data_group


if __name__ == "__main__":
    run_id = "n_10"
    extracted = main(run_id)

    # Create the JSON file spath
    folder = Path(f"{base_path}/{run_id}/_eval/")
    folder.mkdir(parents=True, exist_ok=True)

    # Write the JSON file
    with open(folder / "reviewed.json", 'w') as json_file:
        json.dump(extracted, json_file, indent=4)
