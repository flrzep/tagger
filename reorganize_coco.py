import os
import json
import shutil
from PIL import Image
import numpy as np
import cv2
import re

def reorganize_to_coco(images_dir, output_dir, class_name):
    """
    Reorganize images and annotations into COCO format in a new directory.
    Args:
        images_dir (str): Path to the directory containing images and JSON annotations.
        output_dir (str): Path to the output COCO dataset directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_out = os.path.join(output_dir, "images")
    ann_out = os.path.join(output_dir, "annotations")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    ann_id = 1
    img_id = 1
    categories = {}

    for fname in os.listdir(images_dir):
        if (
            fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
            and len(fname.split('.')) == 2
            and fname.split('.')[0].isdigit()
            and len(fname.split('.')[0]) == 5
        ):
            img_path = os.path.join(images_dir, fname)
            json_path = os.path.join(images_dir, os.path.splitext(fname)[0] + ".json")
            # Copy image
            shutil.copy(img_path, os.path.join(images_out, fname))
            # Get image info
            with Image.open(img_path) as img:
                width, height = img.size
            coco["images"].append({
                "id": img_id,
                "file_name": fname,
                "width": width,
                "height": height
            })
            # Parse annotation
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                object_id = 1
                for obj in data.get("objects", []):
                    cat_name = class_name
                    # cat_name = obj.get("class_name", "object")
                    if cat_name not in categories:
                        categories[cat_name] = len(categories) + 1
                        coco["categories"].append({"id": categories[cat_name], "name": cat_name})
                    cat_id = categories[cat_name]
                    segmentation = []
                    bbox = [0, 0, 0, 0]
                    area = 0
                    # Try to get polygon from mask file if object_id and mask dir exist
                    # object_id = str(obj.get("object_id", ""))[-1]
                    # Find the mask file by matching pattern: {object_id} at the start of the file name
                    mask_file = None
                    mask_dir = None
                    for f in os.listdir(images_dir):
                        match = re.match(rf".*{img_id}_masks", f)
                        if match:
                            mask_dir = os.path.join(images_dir, f)
                            break
                    for f in os.listdir(mask_dir):
                        if f.startswith(f"{object_id}_"):
                            mask_file = os.path.join(mask_dir, f)
                            break   
                        
                    if os.path.isfile(mask_file):
                        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Find contours (external only)
                            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for contour in contours:
                                if len(contour) >= 3:
                                    poly = contour.flatten().tolist()
                                    if len(poly) >= 6:  # at least 3 points
                                        segmentation.append(poly)
                            # Compute bbox and area from mask
                            ys, xs = np.where(mask > 0)
                            if xs.size > 0 and ys.size > 0:
                                x_min, y_min, x_max, y_max = xs.min(), ys.min(), xs.max(), ys.max()
                                bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                                area = int(np.sum(mask > 0))
                    elif obj.get("segmentation_mask"):
                        pts = obj["segmentation_mask"]
                        segmentation = [sum(pts, [])]  # flatten
                        xs = [pt[0] for pt in pts]
                        ys = [pt[1] for pt in pts]
                        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        area = (x_max - x_min) * (y_max - y_min)
                    elif obj.get("screen_position"):
                        x = obj["screen_position"].get("x", 0)
                        y = obj["screen_position"].get("y", 0)
                        bbox = [x, y, 1, 1]
                        area = 1
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "segmentation": segmentation,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id += 1
                    object_id += 1
            img_id += 1
    # Load existing COCO annotation file if it exists
    instances_path = os.path.join(ann_out, "instances.json")
    if os.path.isfile(instances_path):
        with open(instances_path, 'r') as f:
            existing_coco = json.load(f)
        # Merge images, annotations, categories (avoid duplicates)
        existing_img_files = {img['file_name']: img['id'] for img in existing_coco.get('images', [])}
        existing_cat_names = {cat['name']: cat['id'] for cat in existing_coco.get('categories', [])}
        # Update counters
        img_id = max([*existing_img_files.values(), img_id-1]) + 1 if existing_img_files else img_id
        ann_id = max([ann['id'] for ann in existing_coco.get('annotations', [])] + [ann_id-1]) + 1 if existing_coco.get('annotations') else ann_id
        categories = existing_cat_names.copy()
        # Add existing data
        coco['images'].extend(existing_coco.get('images', []))
        coco['annotations'].extend(existing_coco.get('annotations', []))
        coco['categories'].extend(existing_coco.get('categories', []))
    # Save COCO annotation file
    with open(instances_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"COCO dataset created at {output_dir}")

if __name__ == "__main__":
    reorganize_to_coco("images", "coco_dataset", "l_profile")
