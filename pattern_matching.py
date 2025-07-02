import os
import cv2
import numpy as np
import json


def get_pattern_image_path(config_path, option=None):
    """
    Load the pattern image path from a JSON config file based on the selected option.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    if option is None:
        option = config.get("default", "square")
    pattern_file = config["pattern_options"].get(option)
    if not pattern_file:
        raise ValueError(f"Pattern option '{option}' not found in config.")
    return pattern_file


def run_pattern_matching(directory, config_path="pattern_config.json", option=None, threshold=0.8, show_results=True):
    """
    Pattern matching on images in a directory using a pattern image selected from config.
    Args:
        directory (str): Path to the directory containing images to search.
        config_path (str): Path to the JSON config file with pattern options.
        option (str): Which pattern to use (e.g., 'square', 'l_shape').
        threshold (float): Matching threshold (default: 0.8).
        show_results (bool): Whether to display images with matches.
    """
    pattern_image_path = get_pattern_image_path(config_path, option)
    pattern = cv2.imread(pattern_image_path, cv2.IMREAD_GRAYSCALE)
    if pattern is None:
        print(f"Pattern image not found: {pattern_image_path}")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image is not None else None
            if gray is None:
                print(f"Failed to load image: {image_path}")
                continue
            # Perform template matching
            result = cv2.matchTemplate(gray, pattern, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            match_count = len(locations[0])
            if match_count > 0:
                print(f"Pattern found in {filename} at {match_count} location(s).")
                for pt in zip(*locations[::-1]):
                    cv2.rectangle(image, pt, (pt[0] + pattern.shape[1], pt[1] + pattern.shape[0]), (0, 0, 255), 2)
                if show_results:
                    cv2.imshow(f"Matches in {filename}", image)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f"Matches in {filename}")
            else:
                print(f"No match found in {filename}.")
    if show_results:
        cv2.destroyAllWindows()

# Add more helper functions as needed for pattern matching
