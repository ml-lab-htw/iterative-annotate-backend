import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables
load_dotenv()
base_path = os.getenv('BASE_DIR')

# Paths setup
data_path = "test/crop-detection/sub-dataset"
script_path = "test/utils/visualisation/performance"
data_dir = os.path.join(base_path, data_path)

def load_json_file(run_name):
    path = os.path.join(data_dir, f"{run_name}/_eval/reviewed.json")

    with open(path, "r") as f:
        data = json.load(f)

    return data

def extract_loss_data(data):
    images = [entry['training']['images'] for entry in data if entry['training']]
    normalized_loss = [entry['training']['totalChange'] for entry in data if entry['training']]
    return images, normalized_loss

def plot_loss_comparison(data_n25, data_n10):
    plt.figure(figsize=(12, 8))
    plt.tight_layout()

    images_n25, loss_n25 = extract_loss_data(data_n25)
    images_n10, loss_n10 = extract_loss_data(data_n10)

    plt.plot(images_n10, loss_n10, marker='s', label='20 bundle run', color='#028391')
    plt.plot(images_n25, loss_n25, marker='o', label='8 bundle run', color='#E85917')

    plt.xlabel('Images processed')
    plt.ylabel('Loss function change')
    plt.legend()
    plt.grid(True)

    # Set x-axis limits and ticks
    plt.xlim(0, 200)
    plt.xticks(np.arange(0, 225, 25))

    save_path = os.path.join(base_path, script_path, 'pdf_export/loss_change_comparison.pdf')
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()
    plt.close()

    print(f"Loss comparison visualization saved to {save_path}")

def main():
    data_n25 = load_json_file("n_25")
    data_n10 = load_json_file("n_10")
    plot_loss_comparison(data_n25, data_n10)

if __name__ == "__main__":
    main()