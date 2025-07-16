import os
import cv2
import json
import numpy as np
from PIL import Image
import ctypes
import re


def get_image_files(directory):
    """
    Get a list of image files in the specified directory.
    Supported formats: .jpg, .jpeg, .png, .gif, .bmp
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = []

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and re.match(r'^\d{5}\.(jpg|jpeg|png|gif|bmp)$', filename, re.IGNORECASE):
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




def superimpose_colored_mask(image, mask_path):
    """
    Superimpose the mask on the image, coloring each object number with a unique color.
    The mask is expected to be a single-channel image where each pixel value corresponds to an object number.
    """
    if not os.path.isfile(mask_path):
        return image

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[:2] != image.shape[:2]:
        # Try to resize mask if shape doesn't match
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Generate a color map for up to 20 objects (extend as needed)
    color_map = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
        (64, 0, 64), (0, 64, 64), (192, 192, 192), (128, 128, 128)
    ]

    mask_rgb = np.zeros_like(image)
    for obj_nr in np.unique(mask):
        if obj_nr == 0:
            continue  # 0 is background
        color = color_map[int(obj_nr) % len(color_map)]
        mask_rgb[mask == obj_nr] = color

    
    blended = cv2.addWeighted(image, 0.7, mask_rgb, 0.5, 0)
    return blended


def run_object_overlay_viewer(directory, show_combined=False, nr_imgs=5):
    """
    Original main logic from was_main.py: overlays JSON objects and combined masks on images.
    """
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
    
    # get shape of image files
    img_files_length = len(image_files)

    # generate a mask array
    # mask_array = np.zeros((img_files_length, 0), dtype=np.uint8)
    rnd_array = np.random.randint(0, high=img_files_length , size=(img_files_length))

    mask_array = rnd_array < nr_imgs

    image_files = [image_files[i] for i in range(img_files_length) if mask_array[i]]

    if image_files:
        from PIL import Image as PILImage
        
        for image in image_files:
            print(image)

            image_path = os.path.join(directory, image)
            img = cv2.imread(image_path)
            if img is not None:
                # Draw JSON object positions if JSON exists
                json_name = os.path.splitext(image)[0] + ".json"
                json_path = os.path.join(directory, json_name)
                # Blend the mask with a black and white version of the image
                image_bw = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                image_bw = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for blending
                # Use cv2.addWeighted to blend the original image and the mask
                img_with_objects = draw_json_objects(image_bw, json_path)

                # --- Load combined mask if available ---
                mask_superimposed = None
                image_number = os.path.splitext(image)[0]
                combined_mask_name = f"combined_mask_{image_number}.png"
                combined_mask_path = os.path.join(directory, combined_mask_name)
                if os.path.isfile(combined_mask_path):
                    mask_superimposed = superimpose_colored_mask(img_with_objects, combined_mask_path)

                images_to_show = [img]
                window_titles = ['Original']
                #if img_with_objects is not None:
                #    images_to_show.append(img_with_objects)
                #    window_titles.append('Objects')
                if mask_superimposed is not None:
                    images_to_show.append(mask_superimposed)
                    window_titles.append('Combined Mask')

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

                    # Convert OpenCV image to PIL
                    combined_pil = PILImage.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                    combined_pil.show(title=f'Combined - {image}')
                else:
                    # Show each image in a separate window
                    for im, title in zip(images_to_show, window_titles):
                        # Scale if needed
                        h, w = im.shape[:2]
                        scale = min(max_width / w, max_height / h, 1.0)
                        if scale < 1.0:
                            im = cv2.resize(im, (int(w * scale), int(h * scale)))
                        
                        # Convert to PIL and show
                        pil_img = PILImage.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                        pil_img.show(title=f'{title} - {image}')
            
            # Control flow
            input("Press Enter to continue...")
    else:
        print("No image files found in the specified directory.")


def main():
    directory = "C:/Users/FLRZ01/OneDrive - SMS group GmbH/Desktop/dev/tagger/images"
    # Uncomment the function you want to run:
    # run_object_overlay_viewer(directory, show_combined=False)
    pass  # No-op if nothing is uncommented


if __name__ == "__main__":
    main()