import ipywidgets as widgets
from IPython.display import display
import cv2

def plot_boxes_on_image(
    image, result, box_width=2, overlap_mode="color", class_names=None, colors=None, alpha=0.4
):
    import matplotlib.pyplot as plt
    import numpy as np

    # If image is a path, load it
    if isinstance(image, str):
        img_loaded = cv2.imread(image)
        if img_loaded is None:
            raise ValueError(f"Could not load image from path: {image}")
        image = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    # Get boxes, class_ids, and scores from result
    if hasattr(result, "boxes") and hasattr(result.boxes, "xyxy"):
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, "cls") else np.zeros((len(boxes),))
        scores = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else np.ones((len(boxes),))
    else:
        boxes = np.array([])
        class_ids = np.array([])
        scores = np.array([])

    if colors is None:
        num_classes = int(np.max(class_ids) + 1) if len(class_ids) else 1
        colors = plt.cm.get_cmap('tab20', num_classes)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        class_id = int(class_ids[i]) if len(class_ids) else 0
        score = scores[i] if len(scores) else 1.0
        color = colors(class_id) if callable(colors) else colors[class_id % len(colors)]
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=box_width,
            edgecolor=color,
            facecolor=color if overlap_mode == "alpha" else "none",
            alpha=alpha if overlap_mode == "alpha" else 1.0,
            fill=overlap_mode == "alpha"
        )
        ax.add_patch(rect)
        label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
        ax.text(
            x1, y1 - 2, label,
            fontsize=12,
            color='white',
            bbox=dict(facecolor=color, alpha=0.5, pad=0)
        )
    plt.axis('off')
    plt.show()

def plot_boxes_on_image_interactive(
    image, result, box_width=2, overlap_mode="color", class_names=None, colors=None, alpha=0.4
):
    import matplotlib.pyplot as plt
    import numpy as np

    # If image is a path, load it
    if isinstance(image, str):
        img_loaded = cv2.imread(image)
        if img_loaded is None:
            raise ValueError(f"Could not load image from path: {image}")
        image_np = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
    else:
        image_np = image.copy()

    # Get boxes, class_ids, and scores from result
    if hasattr(result, "boxes") and hasattr(result.boxes, "xyxy"):
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, "cls") else np.zeros((len(boxes),))
        scores = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else np.ones((len(boxes),))
    else:
        boxes = np.array([])
        class_ids = np.array([])
        scores = np.array([])

    if colors is None:
        num_classes = int(np.max(class_ids) + 1) if len(class_ids) else 1
        colors = plt.cm.get_cmap('tab20', num_classes)

    def show(idx):
        img_disp = image_np.copy()
        plt.figure(figsize=(10, 10))
        # plt.imshow(img_disp)
        ax = plt.gca()
        if len(boxes) > 0:
            box = boxes[idx]
            x1, y1, x2, y2 = box
            class_id = int(class_ids[idx]) if len(class_ids) else 0
            score = scores[idx] if len(scores) else 1.0
            color = colors(class_id) if callable(colors) else colors[class_id % len(colors)]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=box_width,
                edgecolor=color,
                facecolor=color if overlap_mode == "alpha" else "none",
                alpha=alpha if overlap_mode == "alpha" else 1.0,
                fill=overlap_mode == "alpha"
            )
            ax.add_patch(rect)
            label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
            ax.text(
                x1, y1 - 2, label,
                fontsize=12,
                color='white',
                bbox=dict(facecolor=color, alpha=0.5, pad=0)
            )
        plt.axis('off')
        plt.show()

    import ipywidgets as widgets
    widgets.interact(
        show,
        idx=widgets.IntSlider(
            min=0,
            max=min(4, max(len(boxes)-1, 0)),  # Limit to first 5 objects
            step=1,
            description='Box'
        )
    )

# Usage:
# plot_boxes_on_image(image, boxes)
# plot_boxes_on_image_interactive(list_of_images, list_of_boxes)
