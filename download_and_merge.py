import os
import kagglehub
import shutil

# 1. Download both datasets
print("Downloading Dataset 1...")
path1 = kagglehub.dataset_download("mohamedsayed12/eye-disease-dataset")
print("Downloading Dataset 2...")
path2 = kagglehub.dataset_download("tejpal123/eye-disease-dataset")

# 2. Define the target classes we want
target_classes = ['Bulging_Eyes', 'Cataracts', 'Crossed_Eyes', 'Glaucoma', 'Uveitis']
merged_path = "data/merged_dataset"

# Create folders
for cls in target_classes:
    os.makedirs(os.path.join(merged_path, cls), exist_ok=True)

def merge_folders(source_root, prefix):
    for root, dirs, files in os.walk(source_root):
        folder_name = os.path.basename(root)
        if folder_name in target_classes:
            print(f"Merging {folder_name} from {prefix}...")
            for i, file in enumerate(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src = os.path.join(root, file)
                    # Unique filename to prevent overwriting
                    dst = os.path.join(merged_path, folder_name, f"{prefix}_{i}_{file}")
                    shutil.copy(src, dst)

# Execute merge
merge_folders(path1, "set1")
merge_folders(path2, "set2")

print(f"\n✅ Done! Robust dataset created at: {os.path.abspath(merged_path)}")
for cls in target_classes:
    count = len(os.listdir(os.path.join(merged_path, cls)))
    print(f"Total {cls} images: {count}")