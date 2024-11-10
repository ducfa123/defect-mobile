import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Define source and destination directories
source_dir = 'datasets/datasets'
images_dir = os.path.join(source_dir, 'images')
annotations_dir = os.path.join(source_dir, 'covert_annotations')

# Output directories
output_dirs = {
    'train': {
        'images': 'data/train/images',
        'annotations': 'data/train/covert_annotations'
    },
    'val': {
        'images': 'data/val/images',
        'annotations': 'data/val/covert_annotations'
    },
    'test': {
        'images': 'data/test/images',
        'annotations': 'data/test/covert_annotations'
    }
}

# Create output directories if they don't exist
for split in output_dirs.values():
    os.makedirs(split['images'], exist_ok=True)
    os.makedirs(split['annotations'], exist_ok=True)

# Get list of files in images directory
all_images = os.listdir(images_dir)
all_images = [f for f in all_images if os.path.isfile(os.path.join(images_dir, f))]

# Shuffle the data
random.shuffle(all_images)

# Split data into train, val, and test (8:1:1 ratio)
total_count = len(all_images)
train_count = int(0.8 * total_count)
val_count = int(0.1 * total_count)

train_files = all_images[:train_count]
val_files = all_images[train_count:train_count + val_count]
test_files = all_images[train_count + val_count:]

# Function to copy files
def copy_files(file_list, dest_images_dir, dest_annotations_dir):
    for file_name in file_list:
        # Copy image file
        shutil.copy(os.path.join(images_dir, file_name), os.path.join(dest_images_dir, file_name))
        
        # Copy corresponding annotation file if it exists
        annotation_file = file_name.rsplit('.', 1)[0] + '.txt'  # Assuming annotation files are .txt
        annotation_path = os.path.join(annotations_dir, annotation_file)
        if os.path.exists(annotation_path):
            shutil.copy(annotation_path, os.path.join(dest_annotations_dir, annotation_file))

# Copy files to respective directories
copy_files(train_files, output_dirs['train']['images'], output_dirs['train']['annotations'])
copy_files(val_files, output_dirs['val']['images'], output_dirs['val']['annotations'])
copy_files(test_files, output_dirs['test']['images'], output_dirs['test']['annotations'])

print("Dataset split completed successfully!")