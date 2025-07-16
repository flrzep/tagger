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

def plot_segmentation_on_image(
    image, result, alpha=0.5, class_names=None, colors=None, show_boxes=True, box_width=2
):
    """
    Plot segmentation masks on image with colored overlays for each object.
    
    Args:
        image: Input image (numpy array or path to image)
        result: Model result containing masks, boxes, class_ids, scores
        alpha: Transparency of segmentation overlay (0-1)
        class_names: List of class names for labeling
        colors: Color map or list of colors for different classes
        show_boxes: Whether to also show bounding boxes
        box_width: Width of bounding box lines
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # If image is a path, load it
    if isinstance(image, str):
        img_loaded = cv2.imread(image)
        if img_loaded is None:
            raise ValueError(f"Could not load image from path: {image}")
        image = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)

    # Store original image dimensions
    original_h, original_w = image.shape[:2]
    print(f"Image dimensions: {original_w} x {original_h}")
    
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    ax = plt.gca()

    # Get data from result - handle different result formats
    masks = None
    boxes = np.array([])
    class_ids = np.array([])
    scores = np.array([])

    # Try to extract masks from different possible formats
    if hasattr(result, "masks") and result.masks is not None:
        if hasattr(result.masks, "data"):
            masks = result.masks.data.cpu().numpy()
            print(f"Masks shape: {masks.shape}")
        elif hasattr(result.masks, "xy"):
            # Handle polygon format
            masks = result.masks.xy
    elif isinstance(result, dict):
        # Handle dictionary format (e.g., from custom models)
        if "masks" in result:
            masks = result["masks"]
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            print(f"Masks shape: {masks.shape}")
        if "boxes" in result:
            boxes = result["boxes"]
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
        if "labels" in result:
            class_ids = result["labels"]
            if hasattr(class_ids, "cpu"):
                class_ids = class_ids.cpu().numpy()
        if "scores" in result:
            scores = result["scores"]
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

    # Get boxes, class_ids, and scores if not already extracted
    if len(boxes) == 0 and hasattr(result, "boxes") and hasattr(result.boxes, "xyxy"):
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, "cls") else np.zeros((len(boxes),))
        scores = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else np.ones((len(boxes),))

    print(f"Found {len(boxes)} boxes")

    # Set up colors
    if colors is None:
        num_classes = int(np.max(class_ids) + 1) if len(class_ids) else 1
        colors = plt.cm.get_cmap('tab20', num_classes)
        cmaps = ['Blues', 'Greens', 'Reds', 'Oranges', 'Purples']
    
    # Plot segmentation masks
    if masks is not None:
        for i, mask in enumerate(masks):
            class_id = int(class_ids[i]) if len(class_ids) > i else 0
            score = scores[i] if len(scores) > i else 1.0
            
            # Get color for this class

            nr_same_class = np.sum(class_ids == class_id)
            idx_in_class = np.where(class_ids == class_id)[0].tolist().index(i) if nr_same_class > 0 else 0

            cmap = cmaps[class_id]
            colors = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else colors
           
            color = colors(idx_in_class/nr_same_class)

            # print(f"Using color map: {cmap} for class {class_id} with index {idx_in_class} from cmap position: {idx_in_class/nr_same_class} and color: {color}")

            # Handle different mask formats
            if isinstance(mask, np.ndarray) and mask.ndim == 2:
                # Resize mask to match image dimensions if needed
                if mask.shape != (original_h, original_w):
                    # print(f"Resizing mask {i} from {mask.shape} to ({original_h}, {original_w})")
                    mask = cv2.resize(mask.astype(np.float32), (original_w, original_h), 
                                    interpolation=cv2.INTER_NEAREST)
                
                # Create colored mask overlay
                colored_mask = np.zeros((original_h, original_w, 4))
                colored_mask[mask > 0.5] = [*color[:3], alpha]  # Use full opacity for mask
                
                # Use imshow WITHOUT extent parameter - let matplotlib handle the coordinates
                ax.imshow(colored_mask, alpha=alpha)
                
            elif hasattr(mask, '__len__') and len(mask) > 0:
                # Polygon format
                from matplotlib.patches import Polygon
                if isinstance(mask[0], (list, np.ndarray)):
                    # Multiple polygons
                    for poly in mask:
                        if len(poly) >= 6:  # At least 3 points (x,y pairs)
                            points = np.array(poly).reshape(-1, 2)
                            polygon = Polygon(points, closed=True, alpha=alpha, 
                                            facecolor=color, edgecolor=color*0.5)
                            ax.add_patch(polygon)
                else:
                    # Single polygon
                    if len(mask) >= 6:
                        points = np.array(mask).reshape(-1, 2)
                        polygon = Polygon(points, closed=True, alpha=alpha, 
                                        facecolor=color, edgecolor=color*0.5)
                        ax.add_patch(polygon)

    # Optionally show bounding boxes
    if show_boxes and len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_id = int(class_ids[i]) if len(class_ids) > i else 0
            score = scores[i] if len(scores) > i else 1.0
            
            if callable(colors):
                color = colors(class_id)
            else:
                color = colors[class_id % len(colors)] if isinstance(colors, list) else colors
            
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=box_width,
                edgecolor=color,
                facecolor="none",
                alpha=1.0
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
            ax.text(
                x1, y1 - 2, label,
                fontsize=12,
                color='white',
                bbox=dict(facecolor=color, alpha=0.8, pad=2)
            )

    # IMPORTANT: Set axis limits AFTER adding all elements
    ax.set_xlim(0, original_w)
    ax.set_ylim(original_h, 0)  # Flip y-axis for image coordinates
    
    # Ensure aspect ratio is maintained
    ax.set_aspect('equal')
    
    plt.axis('off')
    plt.title('Segmentation Results')
    plt.tight_layout()
    plt.show()

def plot_boxes_on_image_interactive(
    image, result, box_width=1, overlap_mode="color", class_names=None, colors=None, alpha=0.4, fontsize=10
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
        plt.imshow(img_disp)
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
                fontsize=fontsize,
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
            max=min(50, max(len(boxes)-1, 0)),  # Limit to first 50 objects
            step=1,
            description='Box'
        )
    )

def plot_segmentation_interactive(
    image, result, alpha=0.5, class_names=None, colors=None, show_boxes=True, box_width=2
):
    """
    Interactive segmentation viewer that allows you to toggle individual masks on/off.
    """
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

    # Store original image dimensions
    original_h, original_w = image_np.shape[:2]
    print(f"Original image size: {original_w} x {original_h}")

    # Extract data from result
    masks = None
    boxes = np.array([])
    class_ids = np.array([])
    scores = np.array([])

    # Try to extract masks from different possible formats
    if hasattr(result, "masks") and result.masks is not None:
        if hasattr(result.masks, "data"):
            masks = result.masks.data.cpu().numpy()
            print(f"Mask tensor shape: {masks.shape}")
    elif isinstance(result, dict) and "masks" in result:
        masks = result["masks"]
        if hasattr(masks, "cpu"):
            masks = masks.cpu().numpy()
        print(f"Mask array shape: {masks.shape}")

    # Get other data
    if isinstance(result, dict):
        if "boxes" in result:
            boxes = result["boxes"]
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
        if "labels" in result:
            class_ids = result["labels"]
            if hasattr(class_ids, "cpu"):
                class_ids = class_ids.cpu().numpy()
        if "scores" in result:
            scores = result["scores"]
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
    elif hasattr(result, "boxes"):
        if hasattr(result.boxes, "xyxy"):
            boxes = result.boxes.xyxy.cpu().numpy()
        if hasattr(result.boxes, "cls"):
            class_ids = result.boxes.cls.cpu().numpy()
        if hasattr(result.boxes, "conf"):
            scores = result.boxes.conf.cpu().numpy()

    # Set up colors
    if colors is None:
        num_classes = int(np.max(class_ids) + 1) if len(class_ids) else 1
        colors = plt.cm.get_cmap('tab20', num_classes)

    def show_mask(idx, show_all_masks=False):
        plt.figure(figsize=(15, 15))  # Larger figure size
        # plt.imshow(image_np)
        ax = plt.gca()
        
        if masks is not None:
            if show_all_masks:
                # Show all masks
                for i, mask in enumerate(masks):
                    class_id = int(class_ids[i]) if len(class_ids) > i else 0
                    color = colors(class_id) if callable(colors) else colors[class_id % len(colors)]
                    
                    # Resize mask to match original image if needed
                    if mask.shape != (original_h, original_w):
                        print(f"Resizing mask {i} from {mask.shape} to ({original_h}, {original_w})")
                        mask_resized = cv2.resize(mask.astype(np.float32), (original_w, original_h), 
                                                interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_resized = mask
                    
                    # Create colored overlay
                    colored_mask = np.zeros((original_h, original_w, 4))
                    colored_mask[mask_resized > 0.5] = [*color[:3], alpha]
                    
                    # Overlay the mask
                    ax.imshow(colored_mask, alpha=alpha, extent=[0, original_w, original_h, 0])
            else:
                # Show single mask
                if idx < len(masks):
                    mask = masks[idx]
                    class_id = int(class_ids[idx]) if len(class_ids) > idx else 0
                    color = colors(class_id) if callable(colors) else colors[class_id % len(colors)]
                    
                    print(f"Mask {idx} shape: {mask.shape}, Image shape: ({original_h}, {original_w})")
                    
                    # Resize mask to match original image if needed
                    if mask.shape != (original_h, original_w):
                        print(f"Resizing mask from {mask.shape} to ({original_h}, {original_w})")
                        mask_resized = cv2.resize(mask.astype(np.float32), (original_w, original_h), 
                                                interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_resized = mask
                    
                    # Create colored overlay
                    colored_mask = np.zeros((original_h, original_w, 4))
                    colored_mask[mask_resized > 0.5] = [*color[:3], alpha]
                    
                    # Overlay the mask with proper extent
                    ax.imshow(colored_mask, alpha=alpha, extent=[0, original_w, original_h, 0])
        
        # Show boxes if requested
        if show_boxes and len(boxes) > 0:
            if show_all_masks:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    class_id = int(class_ids[i]) if len(class_ids) > i else 0
                    score = scores[i] if len(scores) > i else 1.0
                    color = colors(class_id) if callable(colors) else colors[class_id % len(colors)]
                    
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=box_width, edgecolor=color, facecolor="none")
                    ax.add_patch(rect)
                    
                    label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
                    ax.text(x1, y1 - 2, label, fontsize=12, color='white',
                           bbox=dict(facecolor=color, alpha=0.8, pad=2))
            else:
                if idx < len(boxes):
                    box = boxes[idx]
                    x1, y1, x2, y2 = box
                    class_id = int(class_ids[idx]) if len(class_ids) > idx else 0
                    score = scores[idx] if len(scores) > idx else 1.0
                    color = colors(class_id) if callable(colors) else colors[class_id % len(colors)]
                    
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=box_width, edgecolor=color, facecolor="none")
                    ax.add_patch(rect)
                    
                    label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
                    ax.text(x1, y1 - 2, label, fontsize=12, color='white',
                           bbox=dict(facecolor=color, alpha=0.8, pad=2))
        
        # Set proper axis limits to prevent cropping
        ax.set_xlim(0, original_w)
        ax.set_ylim(original_h, 0)  # Flip y-axis for image coordinates
        plt.axis('off')
        plt.title(f'Segmentation - {"All Objects" if show_all_masks else f"Object {idx}"}')
        plt.tight_layout()
        plt.show()

    max_objects = len(masks) if masks is not None else len(boxes)
    
    widgets.interact(
        show_mask,
        idx=widgets.IntSlider(min=0, max=max(max_objects-1, 0), step=1, description='Object'),
        show_all_masks=widgets.Checkbox(value=False, description='Show All')
    )

# Usage examples:
# plot_segmentation_on_image(image, result)
# plot_segmentation_interactive(image, result)
