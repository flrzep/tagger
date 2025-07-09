import os
import cv2
import json
import numpy as np
from PIL import Image
import ctypes
import re
from overlay import run_object_overlay_viewer
from pattern_matching import run_pattern_matching


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

    # Blend the mask with the image
    blended = cv2.addWeighted(image, 0.7, mask_rgb, 0.5, 0)
    return blended


def extract_spatial_features(image, max_features=500):
    """
    Extract ORB keypoints and descriptors, and return their spatial (x, y) positions.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if keypoints is None or len(keypoints) == 0:
        return np.empty((0, 2)), [], None
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points, keypoints, descriptors

def match_spatial_features(points1, points2, max_distance=50):
    """
    Match features between two sets of points based on spatial proximity.
    Returns indices of matching points in points1 and points2.
    """
    matches = []
    for i, pt1 in enumerate(points1):
        distances = np.linalg.norm(points2 - pt1, axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] < max_distance:
            matches.append((i, min_idx))
    return matches

def run_progressive_feature_matching(directory, show_combined=False):
    """
    Original main logic from main.py: progressive spatial feature matching between images.
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
    image_files.sort()  # Ensure consistent order

    # --- Feature extraction and progressive matching ---
    spatial_features = []
    gray_images = []
    filenames = []

    if image_files:
        print("Image files found:")
        for image in image_files:
            print(image)
            image_path = os.path.join(directory, image)
            img = cv2.imread(image_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_images.append(gray)
                filenames.append(image)
                points, keypoints, descriptors = extract_spatial_features(gray)
                spatial_features.append({
                    "filename": image,
                    "points": points,
                    "keypoints": keypoints,
                    "descriptors": descriptors
                })
            else:
                print(f"Failed to load image: {image}")

        # Progressive matching: start with first image, improve matching step by step
        if len(spatial_features) >= 2:
            # Start with the first image's features as reference
            ref_points = spatial_features[0]["points"]
            ref_img = cv2.cvtColor(gray_images[0], cv2.COLOR_GRAY2BGR)

            for i in range(1, len(spatial_features)):
                curr_points = spatial_features[i]["points"]
                curr_img = cv2.cvtColor(gray_images[i], cv2.COLOR_GRAY2BGR)

                matches = match_spatial_features(ref_points, curr_points)
                match_img = np.hstack([ref_img, curr_img])

                for idx1, idx2 in matches:
                    pt1 = tuple(np.round(ref_points[idx1]).astype(int))
                    pt2 = tuple(np.round(curr_points[idx2]).astype(int) + np.array([ref_img.shape[1], 0]))
                    cv2.line(match_img, pt1, pt2, (0, 255, 0), 1)
                    cv2.circle(match_img, pt1, 3, (0, 0, 255), -1)
                    cv2.circle(match_img, pt2, 3, (255, 0, 0), -1)

                # Show the result for this step
                h, w = match_img.shape[:2]
                scale = min(max_width / w, max_height / h, 1.0)
                if scale < 1.0:
                    match_img = cv2.resize(match_img, (int(w * scale), int(h * scale)))
                cv2.imshow(f"Progressive Spatial Feature Matches {filenames[0]} -> {filenames[i]}", match_img)

                # Update reference for next step (improve matching)
                ref_points = curr_points
                ref_img = curr_img

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    directory = "C:/Users/FLRZ01/OneDrive - SMS group GmbH/Desktop/dev/tagger/images"
    # Uncomment the function you want to run:
    # run_progressive_feature_matching(directory, show_combined=False)
    run_object_overlay_viewer(directory, show_combined=True)
    # run_pattern_matching(directory, option="l_shape", threshold=0.8)
    pass  # No-op if nothing is uncommented


if __name__ == "__main__":
    main()