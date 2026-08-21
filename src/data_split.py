import os
import shutil
import random

def split_dataset(base_dir, output_base, train_ratio, val_ratio, test_ratio):

    # if the split was already made, skip it
    if os.path.exists(output_base):  
        print(f"Split dataset folder '{output_base}' already exists, skipping split.")
        return
    
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for split in ['train', 'val', 'test']:
        for class_name in classes:
            os.makedirs(os.path.join(output_base, split, class_name), exist_ok=True)

    for class_name in classes:
        src_folder = os.path.join(base_dir, class_name)
        images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
        random.shuffle(images)

        n = len(images)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        n_test = n - n_train - n_val

        train_files = images[:n_train]
        val_files = images[n_train:n_train+n_val]
        test_files = images[n_train+n_val:]

        for file_list, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            for file_name in file_list:
                src = os.path.join(src_folder, file_name)
                dst = os.path.join(output_base, split, class_name, file_name)
                shutil.copy2(src, dst)
                
    print("Dataset split completed!")