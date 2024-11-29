import os
import random
import shutil
import json
from PIL import Image, ImageDraw
from dotenv import load_dotenv

def create_subset_with_labels(full_dataset_path, subset_path, num_images_per_subset, num_subsets):
    image_folder = os.path.join(full_dataset_path, "images")
    label_folder = os.path.join(full_dataset_path, "labels")

    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    subset_data = []

    for subset_idx in range(num_subsets):
        subset_folder = os.path.join(subset_path, f"bundle_{subset_idx+1}")
        os.makedirs(subset_folder, exist_ok=True)

        demo_image_folder = os.path.join(subset_folder, "demo_images")
        os.makedirs(demo_image_folder, exist_ok=True)

        selected_images = random.sample(image_files, num_images_per_subset)

        for image_idx, image_file in enumerate(selected_images):
            image_path = os.path.join(image_folder, image_file)

            original_image_path = os.path.join(subset_folder, f"image_{image_idx+1}.jpg")
            shutil.copy(image_path, original_image_path)

            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_folder, label_file)

            if os.path.exists(label_path):
                with Image.open(image_path) as img:
                    # Create a copy of the image for drawing bounding boxes
                    demo_img = Image.new("RGB", (300, 300), (0, 0, 0))
                    img_copy = img.copy()
                    img_copy.thumbnail((300, 300))
                    demo_img.paste(img_copy, ((300 - img_copy.width) // 2, (300 - img_copy.height) // 2))
                    draw = ImageDraw.Draw(demo_img)

                    with open(label_path, "r") as f:
                        lines = f.readlines()
                        boxes = []
                        labels = []
                        for line in lines:
                            label, x_center, y_center, bbox_width, bbox_height = map(float, line.split())

                            # Convert normalized coordinates to pixel coordinates
                            labels.append(int(label))

                            # Convert normalized coordinates to demo image coordinate system with padding
                            demo_x1 = int((x_center - bbox_width / 2) * img_copy.width) + (300 - img_copy.width) // 2
                            demo_y1 = int((y_center - bbox_height / 2) * img_copy.height) + (300 - img_copy.height) // 2
                            demo_x2 = int((x_center + bbox_width / 2) * img_copy.width) + (300 - img_copy.width) // 2
                            demo_y2 = int((y_center + bbox_height / 2) * img_copy.height) + (300 - img_copy.height) // 2

                            boxes.append([demo_x1, demo_y1, demo_x2, demo_y2])

                            # Draw bounding boxes on the demo image
                            draw.rectangle([(demo_x1, demo_y1), (demo_x2, demo_y2)], outline="red", width=2)

                    demo_image_path = os.path.join(demo_image_folder, f"demo_image_{image_idx+1}.jpg")
                    demo_img.save(demo_image_path)

                subset_data.append({
                    "image": original_image_path,
                    "annotations": boxes
                })

        image_files = [f for f in image_files if f not in selected_images]

    with open(os.path.join(subset_path, "subset_data.json"), "w") as f:
        json.dump(subset_data, f, indent=4)


if __name__ == "__main__":
    load_dotenv()
    base_dir = os.getenv('BASE_DIR')

    sub_dir = "test/crop-detection/"

    full_dataset_path = os.path.join(base_dir, f'{sub_dir}full-dataset')

    create_subset_with_labels(full_dataset_path,  os.path.join(base_dir, f'{sub_dir}sub-dataset/n_10/truth'), 10, 20)
    create_subset_with_labels(full_dataset_path,  os.path.join(base_dir, f'{sub_dir}sub-dataset/n_25/truth'), 25, 8)
    # create_subset_with_labels(full_dataset_path,  os.path.join(base_dir, 'test/sub-dataset/n_20/truth'), 20, 5)