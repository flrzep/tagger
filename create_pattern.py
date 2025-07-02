import cv2
import numpy as np

# Create a simple synthetic pattern (white square on black background)
def create_pattern_image(save_path, size=32):
    """
    Create a simple pattern image (white square) for template matching.
    Args:
        save_path (str): Where to save the pattern image.
        size (int): Size of the square pattern (default: 32x32).
    """
    pattern = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(pattern, (8, 8), (size-8, size-8), 255, -1)  # white square
    cv2.imwrite(save_path, pattern)
    print(f"Pattern image saved to {save_path}")

def create_l_shape_pattern(save_path, size=32, thickness=8):
    """
    Create a pattern image resembling the cross-section of an angle profile steel bar (L shape).
    Args:
        save_path (str): Where to save the pattern image.
        size (int): Size of the square pattern (default: 32x32).
        thickness (int): Thickness of the L arms (default: 8).
    """
    pattern = np.zeros((size, size), dtype=np.uint8)
    # Draw vertical arm
    pattern[:size, :thickness] = 255
    # Draw horizontal arm
    pattern[size-thickness:, :size] = 255
    cv2.imwrite(save_path, pattern)
    print(f"L-shape pattern image saved to {save_path}")

if __name__ == "__main__":
    # create_pattern_image("pattern.png")
    # Uncomment to create an L-shape pattern:
     create_l_shape_pattern("pattern_l.png")
