import os
import cv2

# Your data path and classes (edit as necessary)
data_root = 'UA-dataset'
classes = ['Fist', 'None', 'Other', 'Point', 'Scale']
target_size = (64, 64)  # Set your desired CNN input size here

for cls in classes:
    folder = os.path.join(data_root, cls)
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(img_path, img_resized)
        else:
            print(f"Failed to read {img_path}")