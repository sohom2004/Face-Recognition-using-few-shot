import os
import random
from PIL import Image

def resize_and_select_images_in_subdirectories(main_directory, output_directory, target_count):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through each subdirectory in the main directory
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path):
            # Create a subdirectory in the output directory
            output_subdir = os.path.join(output_directory, subdir)
            os.makedirs(output_subdir, exist_ok=True)

            # List to store paths of all image files in the subdirectory
            all_image_paths = []

            # Get a list of image files in the subdirectory
            for file in os.listdir(subdir_path):
                if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
                    all_image_paths.append(os.path.join(subdir_path, file))

            # Shuffle the list of image paths to randomize selection
            # random.shuffle(all_image_paths)

            # Select the first target_count images
            selected_image_paths = all_image_paths[:target_count]

            # Iterate through the selected images
            for input_path in selected_image_paths:
                # Get the filename from the input path
                file = os.path.basename(input_path)
                output_path = os.path.join(output_subdir, file)

                # Open the image
                with Image.open(input_path) as img:
                    # Calculate the new size (10% of original size)
                    new_size = tuple(int(dim * 0.1) for dim in img.size)
                    # Resize the image with Lanczos resampling
                    # resized_img = img.resize(new_size)
                    # Save the resized image
                    img.save(output_path)
# Example usage
main_directory = './static/faces_2'
output_directory = './static/faces'
target_count = 100

resize_and_select_images_in_subdirectories(main_directory, output_directory, target_count)
print('Done')