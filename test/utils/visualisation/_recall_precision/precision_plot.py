import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# Load environment variables
load_dotenv()
base_path = os.getenv('BASE_DIR')

# Paths setup
data_path = "test/dead-tree/sub-dataset"
script_path = "test/utils/visualisation/_recall_precision"
data_dir = os.path.join(base_path, data_path)


def load_json_file(run_name):
    path = os.path.join(data_dir, f"{run_name}/_eval/inference.json")

    with open(path, "r") as f:
        data = json.load(f)

    return data


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = get_coordinates(box1)
    x3, y3, x4, y4 = get_coordinates(box2)

    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def get_coordinates(box):
    try:
        # Attempt to get 'coordinates' key
        return box['coordinates']
    except (TypeError, KeyError):
        # If box is not a dict or key doesn't exist, assume it's a list
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return box
        else:
            raise ValueError("Box must be a dict with 'coordinates' key or a list/tuple of four integers.")



def generate_color(counter):
    hex_color = f'#{(counter * 34567 % 0xFFFFFF):06X}'
    return hex_color


def plot_precision_recall_curve(data, run_id):
    plt.figure(figsize=(10, 8))
    plt.tight_layout()

    color_counter = 0

    for bundle in data:
        boxes = bundle["boxes"]

        gt_images = boxes["ground_truth"]
        inf_images = boxes["inference"]

        precisions = []
        recalls = []

        print(f"Processing bundle {bundle['bundle_id']}")

        for threshold in np.arange(0.01, 1.01, 0.01):
            TP, FP, FN = 0, 0, 0

            for gt_boxes, pred_boxes in zip(gt_images, inf_images):
                matched_gt = set()
                matched_pred = set()

                for pred_box in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= threshold:
                        if best_gt_idx not in matched_gt:
                            TP += 1
                            matched_gt.add(best_gt_idx)
                            matched_pred.add(tuple(pred_box))
                        else:
                            FP += 1
                    else:
                        FP += 1

                FN += len(gt_boxes) - len(matched_gt)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

            print(
                f"Threshold: {threshold:.2f}, TP: {TP}, FP: {FP}, FN: {FN}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Calculate Average Precision (AP)
        recalls = np.array(recalls)
        precisions = np.array(precisions)
        ap = auc(recalls, precisions)

        color = generate_color(color_counter)
        plt.plot(recalls, precisions, label=f'Bundle {bundle["bundle_id"]} AP: {ap:.3f}', color=color)
        color_counter += 1

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for Run {run_id}')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    #save_path = os.path.join(base_path, script_path, f'pdf_export/precision_recall_{run_id}.pdf')
    #plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def main(run_id):
    data = load_json_file(run_id)
    plot_precision_recall_curve(data, run_id)


if __name__ == "__main__":
    run_id = "n_10"  # or whatever your run_id is
    main(run_id)
