import os
import random
import shutil

# Tỷ lệ dữ liệu train, valid và test
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

# Đường dẫn thư mục nguồn
source_folder = "D:\\temp\\images"
folders = os.listdir(source_folder)

# Danh sách các categories (nhãn)
categories = [folder for folder in folders if os.path.isdir(os.path.join(source_folder, folder))]
categories.sort()
print("Categories:", categories)

# Tạo thư mục đích
target_folder = "D:\\temp\\dataset"
os.makedirs(target_folder, exist_ok=True)

# Hàm chia dữ liệu
def split_data(source, train_dest, valid_dest, test_dest, train_ratio, valid_ratio, test_ratio):
    files = [f for f in os.listdir(source) if os.path.getsize(os.path.join(source, f)) > 0]
    if len(files) == 0:
        print(f"No valid files in {source}, skipping...")
        return

    # Shuffle files and split
    random.shuffle(files)
    train_len = int(len(files) * train_ratio)
    valid_len = int(len(files) * valid_ratio)
    train_set = files[:train_len]
    valid_set = files[train_len:train_len + valid_len]
    test_set = files[train_len + valid_len:]

    # Copy files to target folders
    for filename in train_set:
        shutil.copyfile(os.path.join(source, filename), os.path.join(train_dest, filename))
    for filename in valid_set:
        shutil.copyfile(os.path.join(source, filename), os.path.join(valid_dest, filename))
    for filename in test_set:
        shutil.copyfile(os.path.join(source, filename), os.path.join(test_dest, filename))

    print(f"Copied {len(train_set)} files to {train_dest}, {len(valid_set)} files to {valid_dest}, {len(test_set)} files to {test_dest}")

# Tạo các thư mục train/valid/test và chạy hàm
train_path = os.path.join(target_folder, "train")
valid_path = os.path.join(target_folder, "valid")
test_path = os.path.join(target_folder, "test")
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for category in categories:
    train_dest = os.path.join(train_path, category)
    valid_dest = os.path.join(valid_path, category)
    test_dest = os.path.join(test_path, category)
    source_path = os.path.join(source_folder, category)

    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(valid_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    print(f"Processing category '{category}'...")
    split_data(source_path, train_dest, valid_dest, test_dest, train_ratio, valid_ratio, test_ratio)