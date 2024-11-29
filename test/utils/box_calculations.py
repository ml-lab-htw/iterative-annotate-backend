
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


def evaluate_bundle(ground_truth, reviewed_data):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    iou_list = []

    for gt_image, reviewed_image in zip(ground_truth, reviewed_data):
        gt_annotations = gt_image["annotations"]
        reviewed_annotations = reviewed_image["annotations"]

        matched_boxes = []

        for gt_box in gt_annotations:
            max_iou = 0
            max_box = None

            for reviewed_box in reviewed_annotations:
                iou = calculate_iou(gt_box, reviewed_box["box"])
                if iou > max_iou:
                    max_iou = iou
                    max_box = reviewed_box

            if max_iou >= 0.5:
                true_positives += 1
                matched_boxes.append(max_box)
                iou_list.append(max_iou)  # Store the IoU for TP matches
            else:
                false_negatives += 1

        false_positives += len(reviewed_annotations) - len(matched_boxes)

    total_ground_truth = sum(len(image["annotations"]) for image in ground_truth)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
    average_precision = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "average_precision": average_precision,
        "mean_iou": mean_iou
    }


def extract_bundle_number(bundle_id):
    return int(bundle_id.split('_')[-1])