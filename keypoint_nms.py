import numpy as np


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_keypoint_distance(kp1, kp2):
    """Calculate average distance between corresponding keypoints"""
    distances = []
    min_len = min(len(kp1), len(kp2))
    
    for i in range(min_len):
        if len(kp1[i]) >= 2 and len(kp2[i]) >= 2:
            # Both keypoints are valid
            if len(kp1[i]) >= 3 and len(kp2[i]) >= 3:
                vis1, vis2 = kp1[i][2], kp2[i][2]
                if vis1 > 0 and vis2 > 0:  # Both visible
                    dist = np.linalg.norm(kp1[i][:2] - kp2[i][:2])
                    distances.append(dist)
    
    return np.mean(distances) if distances else float('inf')


def apply_nms_to_detections(keypoints, boxes, scores, 
                           iou_threshold=0.5, 
                           keypoint_distance_threshold=30.0,
                           score_threshold=0.1):
    """
    Apply Non-Maximum Suppression to filter out duplicate detections
    
    Args:
        keypoints: Array of keypoint detections
        boxes: Array of bounding boxes
        scores: Array of confidence scores
        iou_threshold: IoU threshold for box overlap
        keypoint_distance_threshold: Distance threshold for keypoint similarity
        score_threshold: Minimum score threshold
    
    Returns:
        Filtered keypoints, boxes, scores
    """
    if len(scores) == 0:
        return keypoints, boxes, scores
    
    # Convert to numpy arrays if they aren't already
    keypoints = np.array(keypoints)
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Filter by minimum score first
    score_mask = scores >= score_threshold
    if not np.any(score_mask):
        return [], [], []
    
    keypoints = keypoints[score_mask]
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    
    # Sort by score (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    suppressed = np.zeros(len(scores), dtype=bool)
    
    for i in sorted_indices:
        if suppressed[i]:
            continue
        
        keep_indices.append(i)
        
        # Suppress overlapping detections
        for j in sorted_indices:
            if i == j or suppressed[j]:
                continue
            
            # Check bounding box overlap
            iou = calculate_iou(boxes[i], boxes[j])
            
            # Check keypoint similarity
            kp_distance = calculate_keypoint_distance(keypoints[i], keypoints[j])
            
            # Suppress if either IoU or keypoint distance indicates overlap
            if (iou > iou_threshold or 
                kp_distance < keypoint_distance_threshold):
                suppressed[j] = True
    
    # Return filtered results
    filtered_keypoints = keypoints[keep_indices]
    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]
    
    return filtered_keypoints, filtered_boxes, filtered_scores


def apply_advanced_nms(keypoints, boxes, scores, config=None):
    """
    Advanced NMS with configurable parameters and multiple criteria
    """
    if config is None:
        config = {
            'iou_threshold': 0.3,           # Lower = more aggressive suppression
            'keypoint_distance_threshold': 25.0,  # Distance in pixels
            'score_threshold': 0.1,         # Minimum confidence
            'max_detections': 10,           # Maximum detections to keep
        }
    
    if len(scores) == 0:
        return [], [], []
    
    # Convert to numpy
    keypoints = np.array(keypoints)
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Filter by minimum score
    score_mask = scores >= config['score_threshold']
    if not np.any(score_mask):
        return [], [], []
    
    keypoints = keypoints[score_mask]
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    
    # return _apply_soft_nms(keypoints, boxes, scores, config)
    return _apply_standard_nms(keypoints, boxes, scores, config)



def _apply_standard_nms(keypoints, boxes, scores, config):
    """Standard NMS implementation"""
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    suppressed = np.zeros(len(scores), dtype=bool)
    
    for i in sorted_indices:
        if suppressed[i]:
            continue
        
        keep_indices.append(i)
        
        if len(keep_indices) >= config['max_detections']:
            break
        
        # Suppress overlapping detections
        for j in sorted_indices:
            if i == j or suppressed[j]:
                continue
            
            # Multi-criteria suppression
            should_suppress = False
            
            # Criterion 1: Bounding box overlap
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > config['iou_threshold']:
                should_suppress = True
            
            # Criterion 2: Keypoint proximity
            kp_distance = calculate_keypoint_distance(keypoints[i], keypoints[j])
            if kp_distance < config['keypoint_distance_threshold']:
                should_suppress = True
            
            # Criterion 3: Box center proximity (for very close detections)
            center1 = [(boxes[i][0] + boxes[i][2])/2, (boxes[i][1] + boxes[i][3])/2]
            center2 = [(boxes[j][0] + boxes[j][2])/2, (boxes[j][1] + boxes[j][3])/2]
            center_distance = np.linalg.norm(np.array(center1) - np.array(center2))
            
            if center_distance < config['keypoint_distance_threshold'] * 0.8:
                should_suppress = True
            
            if should_suppress:
                suppressed[j] = True
                # print(f"Suppressed detection {j} due to overlap with {i} (IoU: {iou:.2f}, KP Dist: {kp_distance:.2f}, Center Dist: {center_distance:.2f})")
    
    return keypoints[keep_indices], boxes[keep_indices], scores[keep_indices]


'''

def _apply_soft_nms(keypoints, boxes, scores, config):
    """Soft NMS: reduce scores of overlapping detections instead of removing them"""
    scores = scores.copy()  # Don't modify original
    sorted_indices = np.argsort(scores)[::-1]
    
    for i in range(len(sorted_indices)):
        if scores[sorted_indices[i]] < config['score_threshold']:
            continue
        
        for j in range(i + 1, len(sorted_indices)):
            if scores[sorted_indices[j]] < config['score_threshold']:
                continue
            
            idx_i, idx_j = sorted_indices[i], sorted_indices[j]
            
            # Calculate overlap
            iou = calculate_iou(boxes[idx_i], boxes[idx_j])
            kp_distance = calculate_keypoint_distance(keypoints[idx_i], keypoints[idx_j])
            
            # Apply soft suppression based on overlap
            if iou > config['iou_threshold'] or kp_distance < config['keypoint_distance_threshold']:
                # Gaussian decay based on overlap
                overlap_factor = max(iou, 1.0 - (kp_distance / config['keypoint_distance_threshold']))
                decay = np.exp(-(overlap_factor ** 2) / config['sigma'])
                scores[idx_j] *= decay
    
    # Keep only detections above threshold and limit count
    valid_mask = scores >= config['score_threshold']
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > config['max_detections']:
        # Keep top N by score
        top_indices = np.argsort(scores[valid_indices])[::-1][:config['max_detections']]
        valid_indices = valid_indices[top_indices]
    
    return keypoints[valid_indices], boxes[valid_indices], scores[valid_indices]

'''

def apply_nms_to_cached_predictions(cached_predictions, nms_config):
    """Apply NMS to already cached predictions"""
    for prediction in cached_predictions:
        if len(prediction['scores']) > 0:
            keypoints, boxes, scores = apply_advanced_nms(
                prediction['keypoints'], 
                prediction['boxes'], 
                prediction['scores'], 
                nms_config
            )
            prediction['keypoints'] = keypoints
            prediction['boxes'] = boxes
            prediction['scores'] = scores
    
    return cached_predictions



  