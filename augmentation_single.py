import os
import cv2
import numpy as np

def augment_image(image):
    # Define your augmentations here
    # Example augmentations: rotation, flipping, brightness adjustment

    # Rotation
    def rotate_image(image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    # Horizontal flip
    flipped = cv2.flip(image, 1)

    # Brightness adjustment
    def adjust_brightness(image, factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Zoom
    def zoom_image(image, zoom_factor=1.2):
        height, width = image.shape[:2]
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        start_row, start_col = int((new_height - height) / 2), int((new_width - width) / 2)
        zoomed_image = resized_image[start_row:start_row + height, start_col:start_col + width]
        return zoomed_image

    # Grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)  # Convert back to BGR format

    # Invert
    inverted = cv2.bitwise_not(image)

    # Color changes (e.g., increasing/decreasing red channel)
    def change_color(image, channel, factor):
        modified_image = image.copy()
        modified_image[:, :, channel] = cv2.multiply(modified_image[:, :, channel], factor)
        return modified_image

    augmented_images = [
        rotate_image(image, 15),
        rotate_image(image, -15),
        flipped,
        adjust_brightness(image, 1.2),
        adjust_brightness(image, 0.8),
        zoom_image(image, 1.2),
        zoom_image(image, 0.8),
        grayscale,
        inverted,
        change_color(image, 0, 1.2),  # Increase blue channel
        change_color(image, 1, 1.2),  # Increase green channel
        change_color(image, 2, 1.2),  # Increase red channel
        change_color(image, 0, 0.8),  # Decrease blue channel
        change_color(image, 1, 0.8),  # Decrease green channel
        change_color(image, 2, 0.8)   # Decrease red channel
    ]

    return augmented_images

def augment_images_in_directory(directory):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(supported_formats):
                filepath = os.path.join(root, filename)
                image = cv2.imread(filepath)

                if image is not None:
                    augmented_images = augment_image(image)

                    for i, augmented_image in enumerate(augmented_images):
                        new_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                        new_filepath = os.path.join(root, new_filename)
                        cv2.imwrite(new_filepath, augmented_image)
                    