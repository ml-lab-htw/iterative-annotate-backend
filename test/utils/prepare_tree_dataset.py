import os
import random
import shutil
import json
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

def create_subset_with_labels(full_dataset_path, subset_path, num_images_per_subset, num_subsets):
    image_folder = os.path.join(full_dataset_path, "")
    label_folder = os.path.join(full_dataset_path, "")

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    total_images = len(image_files)

    if total_images < num_images_per_subset:
        print(f"Warning: Not enough images in the dataset. Using all {total_images} images.")
        num_images_per_subset = total_images

    max_possible_subsets = total_images // num_images_per_subset
    if max_possible_subsets < num_subsets:
        print(f"Warning: Can only create {max_possible_subsets} subsets with {num_images_per_subset} images each.")
        num_subsets = max_possible_subsets

    subset_data = []

    for subset_idx in range(num_subsets):
        subset_folder = os.path.join(subset_path, f"bundle_{subset_idx+1}")
        os.makedirs(subset_folder, exist_ok=True)

        demo_image_folder = os.path.join(subset_folder, "demo_images")
        os.makedirs(demo_image_folder, exist_ok=True)

        if len(image_files) < num_images_per_subset:
            print(f"Warning: Not enough images left for subset {subset_idx+1}. Using remaining {len(image_files)} images.")
            selected_images = image_files
        else:
            selected_images = random.sample(image_files, num_images_per_subset)

        for image_idx, image_file in enumerate(selected_images):
            image_path = os.path.join(image_folder, image_file)

            original_image_path = os.path.join(subset_folder, f"image_{image_idx+1}{os.path.splitext(image_file)[1]}")
            shutil.copy(image_path, original_image_path)

            xml_file = os.path.splitext(image_file)[0] + ".xml"
            xml_path = os.path.join(label_folder, xml_file)

            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Get image size from XML
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)

                with Image.open(image_path) as img:
                    # Create a copy of the image for drawing bounding boxes
                    demo_img = Image.new("RGB", (300, 300), (0, 0, 0))
                    img_copy = img.copy()
                    img_copy.thumbnail((300, 300))
                    demo_img.paste(img_copy, ((300 - img_copy.width) // 2, (300 - img_copy.height) // 2))
                    draw = ImageDraw.Draw(demo_img)

                    boxes = []
                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        bndbox = obj.find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)

                        # Convert coordinates to demo image coordinate system with padding
                        scale_x = img_copy.width / img_width
                        scale_y = img_copy.height / img_height
                        demo_x1 = int(xmin * scale_x) + (300 - img_copy.width) // 2
                        demo_y1 = int(ymin * scale_y) + (300 - img_copy.height) // 2
                        demo_x2 = int(xmax * scale_x) + (300 - img_copy.width) // 2
                        demo_y2 = int(ymax * scale_y) + (300 - img_copy.height) // 2

                        boxes.append({
                            "label": name,
                            "coordinates": [demo_x1, demo_y1, demo_x2, demo_y2]
                        })

                        # Draw bounding boxes on the demo image
                        draw.rectangle([(demo_x1, demo_y1), (demo_x2, demo_y2)], outline="red", width=2)
                        draw.text((demo_x1, demo_y1 - 10), name, fill="red")

                    demo_image_path = os.path.join(demo_image_folder, f"demo_image_{image_idx+1}.jpg")
                    demo_img.save(demo_image_path)

                subset_data.append({
                    "image": original_image_path,
                    "annotations": boxes
                })

        image_files = [f for f in image_files if f not in selected_images]

    with open(os.path.join(subset_path, "subset_data.json"), "w") as f:
        json.dump(subset_data, f, indent=4)

    print(f"Created {num_subsets} subsets with {num_images_per_subset} images each.")

if __name__ == "__main__":
    load_dotenv()
    base_dir = os.getenv('BASE_DIR')

    sub_dir = "test/dead-tree/"

    full_dataset_path = os.path.join(base_dir, f'{sub_dir}full-dataset')

    create_subset_with_labels(full_dataset_path, os.path.join(base_dir, f'{sub_dir}sub-dataset/n_10/truth'), num_images_per_subset=10, num_subsets=10)
    # create_subset_with_labels(full_dataset_path, os.path.join(base_dir, f'{sub_dir}sub-dataset/n_20/truth'), num_images_per_subset=20, num_subsets=5)