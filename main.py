import os
import cv2
import json
import numpy as np
from PIL import Image
import ctypes


def get_image_files(directory):
    """
    Get a list of image files in the specified directory.
    Supported formats: .jpg, .jpeg, .png, .gif, .bmp
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = []

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(filename)

    return image_files



def draw_json_objects(image, json_path):
    """
    Draw the position of each object from the JSON file using pixel x and y, and their name.
    """
    if not os.path.isfile(json_path):
        return image  # No JSON file, return image unchanged

    with open(json_path, 'r') as f:
        data = json.load(f)
    objects = data.get("objects", [])

    for obj in objects:
        # Try to get a name for the object
        obj_name = obj.get("object_id", obj.get("class_name", "object"))
        # Get pixel position if available
        screen_pos = obj.get("screen_position", {})
        if screen_pos and "x" in screen_pos and "y" in screen_pos:
            x = int(round(screen_pos["x"]))
            y = int(round(screen_pos["y"]))
            cv2.circle(image, (x, y), 6, (0, 0, 255), 2)  # Red circle
            cv2.putText(
                image,
                str(obj_name),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
        # Draw segmentation mask if available and not null
        mask = obj.get("segmentation_mask", None)
        if mask:
            pts = np.array(mask, dtype=np.int32).reshape((-1, 2))
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
            alpha = 0.3
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    return image



def combine_masks_and_superimpose(main_image_path, mask_paths):
    """
    Combine all mask images into one color mask and superimpose on the main image in grayscale.
    Returns the superimposed image as a numpy array.
    """
    # Assign a unique color for each mask (cycling through a palette)
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 128, 128), (255, 128, 0), (128, 0, 255), (0, 128, 255)
    ]
    combined = None
    for idx, mask_path in enumerate(mask_paths):
        mask = Image.open(mask_path).convert("L")
        color = palette[idx % len(palette)]
        color_mask = Image.new("RGBA", mask.size, color + (0,))
        color_mask.putalpha(mask)
        if combined is None:
            combined = Image.new("RGBA", mask.size, (0, 0, 0, 0))
        combined = Image.alpha_composite(combined, color_mask)

    # Superimpose on main image in grayscale
    main_img = Image.open(main_image_path).convert("L").convert("RGBA")
    superimposed = Image.alpha_composite(main_img, combined)
    # Convert to OpenCV format for display
    superimposed_cv = cv2.cvtColor(np.array(superimposed), cv2.COLOR_RGBA2BGR)
    return superimposed_cv



def main():
    directory = "C:/Users/FLRZ01/OneDrive - SMS group GmbH/Desktop/dev/tagger/images"
    mask_dir = os.path.join(directory, "mask")
    show_combined = False  # Set to False to show images in separate windows

    if not os.path.isdir(directory):
        print("The specified path is not a valid directory.")
        return

    print(os.path.abspath(directory))

    # Get screen size (Windows)
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    max_width = int(screen_width * 0.95)
    max_height = int(screen_height * 0.95)

    image_files = get_image_files(directory)

    if image_files:
        print("Image files found:")
        for image in image_files:
            print(image)

            image_path = os.path.join(directory, image)
            img = cv2.imread(image_path)
            if img is not None:
                # Draw JSON object positions if JSON exists
                json_name = os.path.splitext(image)[0] + ".json"
                json_path = os.path.join(directory, json_name)
                img_with_objects = draw_json_objects(img.copy(), json_path)

                # --- Combine masks and superimpose ---
                mask_prefix = os.path.splitext(image)[0]
                mask_subdir = os.path.join(directory, f"{mask_prefix}_masks")
                mask_superimposed = None
                if os.path.isdir(mask_subdir):
                    mask_paths = [
                        os.path.join(mask_subdir, f)
                        for f in os.listdir(mask_subdir)
                        if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg'}
                    ]
                    if mask_paths:
                        mask_superimposed = combine_masks_and_superimpose(image_path, mask_paths)

                images_to_show = [img]
                window_titles = ['Original']
                if img_with_objects is not None:
                    images_to_show.append(img_with_objects)
                    window_titles.append('Objects')
                if mask_superimposed is not None:
                    images_to_show.append(mask_superimposed)
                    window_titles.append('Masks')

                if show_combined:
                    # Resize all images to the same width for stacking vertically
                    min_width = min(im.shape[1] for im in images_to_show)
                    resized_images = [
                        cv2.resize(im, (min_width, int(im.shape[0] * min_width / im.shape[1])))
                        for im in images_to_show
                    ]
                    combined = cv2.vconcat(resized_images)

                    # Scale down if too large for the screen
                    h, w = combined.shape[:2]
                    scale = min(max_width / w, max_height / h, 1.0)
                    if scale < 1.0:
                        combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

                    cv2.imshow(f'Combined - {image}', combined)
                else:
                    # Show each image in a separate window
                    for im, title in zip(images_to_show, window_titles):
                        # Scale down if too large for the screen
                        h, w = im.shape[:2]
                        scale = min(max_width / w, max_height / h, 1.0)
                        if scale < 1.0:
                            im = cv2.resize(im, (int(w * scale), int(h * scale)))
                        cv2.imshow(f'{title} - {image}', im)
            else:
                print(f"Failed to load image: {image}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No image files found in the specified directory.")


if __name__ == "__main__":
    main()