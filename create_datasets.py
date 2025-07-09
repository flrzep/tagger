import os
import json
import shutil
from PIL import Image
import numpy as np
import cv2
import re
from tqdm import tqdm

def export_dataset(images_dir, output_dir, format="coco", split_ratio=0.8, num_keypoints=1):
    """
    Export dataset in various formats: coco, yolo, coco_keypoint, yolo_keypoint, labelme_keypoint.
    Args:
        images_dir (str): Path to the directory containing images and JSON annotations.
        output_dir (str): Path to the output dataset directory.
        format (str): One of 'coco', 'yolo', 'coco_keypoint', 'yolo_keypoint', 'labelme_keypoint'.
        split_ratio (float): Ratio of images to use for training (rest for validation, for YOLO formats).
        num_keypoints (int): Number of keypoints per object (for keypoint formats).
    """
    import random
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
                   and len(f.split('.')) == 2 and f.split('.')[0].isdigit() and len(f.split('.')[0]) == 5]
    image_files.sort()

    # Gather all unique class names
    class_names = set()
    for fname in image_files:
        json_path = os.path.join(images_dir, os.path.splitext(fname)[0] + ".json")
        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            for obj in data.get("objects", []):
                class_names.add(obj.get("class_name", "object"))
    class_names = sorted(list(class_names))
    categories = {name: idx for idx, name in enumerate(class_names)}

    # Helper: robust bbox/area/segmentation logic (from COCO)
    def get_bbox_area_segmentation(obj, img_id, object_id, images_dir):
        segmentation = []
        bbox = [0, 0, 0, 0]
        area = 0
        mask_file = None
        mask_dir = None
        for f in os.listdir(images_dir):
            match = re.match(rf".*{img_id}_masks", f)
            if match:
                mask_dir = os.path.join(images_dir, f)
                break
        if mask_dir:
            for f in os.listdir(mask_dir):
                if f.startswith(f"{object_id}_"):
                    mask_file = os.path.join(mask_dir, f)
                    break
        if mask_file and os.path.isfile(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) >= 3:
                        poly = contour.flatten().tolist()
                        if len(poly) >= 6:
                            segmentation.append(poly)
                ys, xs = np.where(mask > 0)
                if xs.size > 0 and ys.size > 0:
                    x_min, y_min, x_max, y_max = xs.min(), ys.min(), xs.max(), ys.max()
                    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                    area = int(np.sum(mask > 0))
        elif obj.get("segmentation_mask"):
            pts = obj["segmentation_mask"]
            segmentation = [sum(pts, [])]
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
        return bbox, area, segmentation

    if format == "coco":
        images_out = os.path.join(output_dir, "images")
        ann_out = os.path.join(output_dir, "annotations")
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(ann_out, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": []}
        ann_id = 1
        img_id = 1
        for fname in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(images_dir, fname)
            json_path = os.path.join(images_dir, os.path.splitext(fname)[0] + ".json")
            shutil.copy(img_path, os.path.join(images_out, fname))
            with Image.open(img_path) as img:
                width, height = img.size
            coco["images"].append({"id": img_id, "file_name": fname, "width": width, "height": height})
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                object_id = 1
                for obj in data.get("objects", []):
                    cat_name = obj.get("class_name", "object")
                    if cat_name not in categories:
                        categories[cat_name] = len(categories) + 1
                        coco["categories"].append({"id": categories[cat_name], "name": cat_name})
                    cat_id = categories[cat_name]
                    bbox, area, segmentation = get_bbox_area_segmentation(obj, img_id, object_id, images_dir)
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
        coco["categories"] = [{"id": idx+1, "name": name} for idx, name in enumerate(class_names)]
        instances_path = os.path.join(ann_out, "instances.json")
        with open(instances_path, 'w') as f:
            json.dump(coco, f, indent=2)
        print(f"COCO dataset created at {output_dir}")

    elif format == "yolo":
        train_images_out = os.path.join(output_dir, "images", "train")
        val_images_out = os.path.join(output_dir, "images", "val")
        train_labels_out = os.path.join(output_dir, "labels", "train")
        val_labels_out = os.path.join(output_dir, "labels", "val")
        for d in [train_images_out, val_images_out, train_labels_out, val_labels_out]:
            os.makedirs(d, exist_ok=True)
        split_idx = int(len(image_files) * split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        def process_image(fname, images_out, labels_out):
            img_path = os.path.join(images_dir, fname)
            json_path = os.path.join(images_dir, os.path.splitext(fname)[0] + ".json")
            shutil.copy(img_path, os.path.join(images_out, fname))
            with Image.open(img_path) as img:
                width, height = img.size
            label_lines = []
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                object_id = 1
                for obj in data.get("objects", []):
                    class_name = obj.get("class_name", "object")
                    cat_id = categories[class_name]
                    bbox, area, segmentation = get_bbox_area_segmentation(obj, fname.split('.')[0], object_id, images_dir)
                    # YOLO bbox: x_center, y_center, w, h (normalized)
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        w_norm = w / width
                        h_norm = h / height
                        label_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                    object_id += 1
            label_path = os.path.join(labels_out, os.path.splitext(fname)[0] + ".txt")
            with open(label_path, 'w') as f:
                f.write("\n".join(label_lines))
        for fname in tqdm(train_files, desc="Processing train images"):
            process_image(fname, train_images_out, train_labels_out)
        for fname in tqdm(val_files, desc="Processing val images"):
            process_image(fname, val_images_out, val_labels_out)
        print(f"YOLO dataset created at {output_dir}")
        print("Found categories (edit as needed):", class_names)

    elif format == "coco_keypoint":
        images_out = os.path.join(output_dir, "images")
        ann_out = os.path.join(output_dir, "annotations")
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(ann_out, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": []}
        ann_id = 1
        img_id = 1
        keypoints_list = ["object_center"]
        coco["categories"].append({
            "id": 1,
            "name": "object",
            "supercategory": "object",
            "keypoints": keypoints_list,
            "skeleton": []
        })
        for fname in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(images_dir, fname)
            json_path = os.path.join(images_dir, os.path.splitext(fname)[0] + ".json")
            shutil.copy(img_path, os.path.join(images_out, fname))
            with Image.open(img_path) as img:
                width, height = img.size
            coco["images"].append({"id": img_id, "file_name": fname, "width": width, "height": height})
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                object_id = 1
                for obj in data.get("objects", []):
                    kp = obj.get("screen_position", None)
                    if kp and "x" in kp and "y" in kp:
                        keypoints = [kp["x"], kp["y"], 2]
                        num_kp = 1
                    else:
                        keypoints = [0, 0, 0]
                        num_kp = 0
                    bbox, area, segmentation = get_bbox_area_segmentation(obj, img_id, object_id, images_dir)
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "keypoints": keypoints,
                        "num_keypoints": num_kp,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id += 1
                    object_id += 1
            img_id += 1
        instances_path = os.path.join(ann_out, "instances_keypoints.json")
        with open(instances_path, 'w') as f:
            json.dump(coco, f, indent=2)
        print(f"COCO keypoint dataset created at {output_dir}")
        print("Keypoint names (edit as needed):", keypoints_list)

    elif format == "yolo_keypoint":
        train_images_out = os.path.join(output_dir, "images", "train")
        val_images_out = os.path.join(output_dir, "images", "val")
        train_labels_out = os.path.join(output_dir, "labels", "train")
        val_labels_out = os.path.join(output_dir, "labels", "val")
        for d in [train_images_out, val_images_out, train_labels_out, val_labels_out]:
            os.makedirs(d, exist_ok=True)
        split_idx = int(len(image_files) * split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        def process_image(fname, images_out, labels_out):
            img_path = os.path.join(images_dir, fname)
            json_path = os.path.join(images_dir, os.path.splitext(fname)[0] + ".json")
            shutil.copy(img_path, os.path.join(images_out, fname))
            with Image.open(img_path) as img:
                width, height = img.size
            label_lines = []
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                object_id = 1
                for obj in data.get("objects", []):
                    class_name = obj.get("class_name", "object")
                    cat_id = categories[class_name]
                    kp = obj.get("screen_position", None)
                    if kp and "x" in kp and "y" in kp:
                        x_kp = kp["x"] / width
                        y_kp = kp["y"] / height
                        v = 2
                    else:
                        x_kp, y_kp, v = 0.0, 0.0, 0
                    bbox, area, segmentation = get_bbox_area_segmentation(obj, fname.split('.')[0], object_id, images_dir)
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        w_norm = w / width
                        h_norm = h / height
                    else:
                        x_center = y_center = w_norm = h_norm = 0.0
                    label_line = f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {x_kp:.6f} {y_kp:.6f} {v}"
                    label_lines.append(label_line)
                    object_id += 1
            label_path = os.path.join(labels_out, os.path.splitext(fname)[0] + ".txt")
            with open(label_path, 'w') as f:
                f.write("\n".join(label_lines))
        for fname in tqdm(train_files, desc="Processing train images"):
            process_image(fname, train_images_out, train_labels_out)
        for fname in tqdm(val_files, desc="Processing val images"):
            process_image(fname, val_images_out, val_labels_out)
        print(f"YOLO keypoint dataset created at {output_dir}")
        print("Found categories (edit as needed):", class_names)

    elif format == "labelme_keypoint":
        os.makedirs(output_dir, exist_ok=True)
        for fname in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(images_dir, fname)
            json_path = os.path.join(images_dir, os.path.splitext(fname)[0] + ".json")
            if not os.path.isfile(json_path):
                continue
            with Image.open(img_path) as img:
                width, height = img.size
            with open(json_path, 'r') as f:
                data = json.load(f)
            shapes = []
            for obj in data.get("objects", []):
                label = obj.get("class_name", "object")
                kp = obj.get("screen_position", None)
                if kp and "x" in kp and "y" in kp:
                    points = [[float(kp["x"]), float(kp["y"])] ]
                    shapes.append({
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "description": "",
                        "shape_type": "point",
                        "flags": {}
                    })
            labelme_json = {
                "version": "5.3.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": fname,
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width
            }
            out_json_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".json")
            with open(out_json_path, 'w') as f:
                json.dump(labelme_json, f, indent=2)
        print(f"LabelMe-style keypoint JSONs exported to {output_dir}")
    else:
        raise ValueError(f"Unknown format: {format}")

if __name__ == "__main__":
    # Example usage:
    # export_dataset("images", "bars_yolo_dataset", format="yolo")
    # export_dataset("images", "bars_coco_dataset", format="coco")
    export_dataset("images", "bars_keypoint_dataset", format="coco_keypoint")
    # export_dataset("images", "bars_keypoint_yolo_dataset", format="yolo_keypoint")
    # export_dataset("images", "labelme_keypoint_jsons", format="labelme_keypoint")
    pass
