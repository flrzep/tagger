import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from inference_utils import get_model
from scipy.ndimage import gaussian_filter1d
import time
import collections
import matplotlib.pyplot as plt
import glob
from keypoint_kalman import KeypointKalmanFilter

import keypoint_nms as kp_nms


import pickle
import hashlib

# Add after the existing imports

def get_file_hash(file_path):
    """Get MD5 hash of a file for change detection"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return None

def get_cache_filename(video_path, conf_threshold):
    """Generate cache filename based on video path and confidence threshold"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = "inference_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"{cache_dir}/{base_name}_conf{conf_threshold:.2f}_predictions.pkl"
    return cache_filename

def save_predictions_cache(video_path, conf_threshold, predictions_data):
    """Save predictions to cache file"""
    cache_filename = get_cache_filename(video_path, conf_threshold)
    
    cache_data = {
        'video_path': video_path,
        'video_hash': get_file_hash(video_path),
        'conf_threshold': conf_threshold,
        'predictions': predictions_data,
        'timestamp': time.time()
    }
    
    try:
        with open(cache_filename, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Predictions cached to: {cache_filename}")
        return True
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")
        return False

def load_predictions_cache(video_path, conf_threshold):
    """Load predictions from cache if valid"""
    cache_filename = get_cache_filename(video_path, conf_threshold)
    
    if not os.path.exists(cache_filename):
        return None
    
    try:
        with open(cache_filename, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validate cache
        current_hash = get_file_hash(video_path)
        if (cache_data.get('video_hash') == current_hash and 
            cache_data.get('conf_threshold') == conf_threshold and
            cache_data.get('video_path') == video_path):
            
            print(f"Loading cached predictions from: {cache_filename}")
            print(f"Cache created: {time.ctime(cache_data.get('timestamp', 0))}")
            return cache_data['predictions']
        else:
            print(f"Cache invalid for {video_path} - will regenerate")
            return None

    except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return None
    

def run_inference_with_cache(video_path, conf_threshold=0.5, apply_nms=True, force_rerun=False):
    """
    Run inference on video with caching support
    
    Args:
        video_path: Path to input video
        conf_threshold: Confidence threshold for detections
        force_rerun: If True, ignore cache and rerun inference
    
    Returns:
        List of predictions for each frame
    """
    
    # Check cache first (unless forced to rerun)
    if not force_rerun:
        cached_predictions = load_predictions_cache(video_path, conf_threshold)
        if cached_predictions is not None:
            if apply_nms:
                return kp_nms.apply_nms_to_cached_predictions(cached_predictions, nms_config)
            return cached_predictions
    
    # If cache is invalid or forced to rerun, run inference
    model, device = _get_model()
    
    print(f"Running inference on {video_path}...")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"NMS enabled: {apply_nms}")

    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames at {fps} FPS")
    print(f"Device: {device}")
    
    # Store all predictions
    all_predictions = []
    inference_times = []
    nms_times = []

    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame_tensor, frame_rgb = preprocess_frame(frame)
            
            # Run inference with timing
            inference_start = time.time()
            with torch.no_grad():
                result = model([frame_tensor.to(device)])[0]
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Convert tensors to numpy and filter by confidence
            scores = result['scores'].cpu().numpy()
            high_conf_mask = scores > conf_threshold
            
            keypoints = result['keypoints'][high_conf_mask].cpu().numpy()
            boxes = result['boxes'][high_conf_mask].cpu().numpy()
            scores = scores[high_conf_mask]
            
            # Apply NMS if enabled
            if apply_nms and len(scores) > 0:
                nms_start = time.time()
                keypoints, boxes, scores = kp_nms.apply_advanced_nms(keypoints, boxes, scores, nms_config)
                nms_time = time.time() - nms_start
                nms_times.append(nms_time)

            # Store prediction data
            frame_prediction = {
                'frame_number': frame_count,
                'keypoints': keypoints,
                'boxes': boxes,
                'scores': scores,
                'inference_time': inference_time
            }
            
            all_predictions.append(frame_prediction)
            
            frame_count += 1
            if frame_count % 10 == 0:  # Progress update every 10 frames
                progress = (frame_count / total_frames) * 100
                avg_inference = np.mean(inference_times[-10:]) * 1000
                avg_nms = np.mean(nms_times[-10:]) * 1000 if nms_times else 0.0
                estimated_remaining = (total_frames - frame_count) * np.mean(inference_times[-10:])
                print(f"Inference progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                      f"Avg: {avg_inference:.1f}ms/frame | "
                      f"NMS: {avg_nms:.1f}ms | "
                      f"ETA: {estimated_remaining:.1f}s")
    
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    
    finally:
        cap.release()
    
    # Print inference statistics
    if inference_times:
        avg_inference = np.mean(inference_times) * 1000
        total_inference_time = sum(inference_times)
        fps_inference = len(inference_times) / total_inference_time
        
        print(f"\nInference complete!")
        print(f"Processed {len(all_predictions)} frames")
        print(f"Average inference time: {avg_inference:.1f}ms")
        print(f"Inference FPS: {fps_inference:.1f}")
        print(f"Total inference time: {total_inference_time:.2f}s")
    
    # Save to cache
    save_predictions_cache(video_path, conf_threshold, all_predictions)
    
    return all_predictions


def _get_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_keypoints=1, num_classes=3) # classes 0 is background
    model.load_state_dict(torch.load('keypointsrcnn_weights_v3_2.pth', map_location=device))
    model.to(device)
    # model.eval() 

    # Add these optimizations after loading the model
    # ----------------------------------- Optimization Test -----------------------------------------

    # Enable mixed precision (AMP) for faster inference
    from torch.cuda.amp import autocast
    use_amp = torch.cuda.is_available()

    # Enable torch.compile (PyTorch 2.0+) for significant speedup
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model, mode="reduce-overhead")
    #     print("Model compiled for faster inference")

    # Set model to eval mode and disable gradients globally
    model.eval()
    torch.set_grad_enabled(False)

    # Optimize CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        print("CUDA optimizations enabled")

    return model, device #, use_amp 
# --------------------------------------------------------------------------------------------------

class TemporalFilter:
    def __init__(self, window_size=5, min_detections=3):
        self.window_size = window_size
        self.min_detections = min_detections
        self.keypoint_history = collections.deque(maxlen=window_size)
        self.bbox_history = collections.deque(maxlen=window_size)
        self.score_history = collections.deque(maxlen=window_size)
        
    def add_frame(self, result, conf_threshold=0.5):
        """Add a frame's detections to the temporal buffer"""
        # Filter by confidence
        scores = result['scores'].cpu().numpy()
        high_conf_mask = scores > conf_threshold
        
        if np.any(high_conf_mask):
            keypoints = result['keypoints'][high_conf_mask].cpu().numpy()
            boxes = result['boxes'][high_conf_mask].cpu().numpy()
            scores = scores[high_conf_mask]
            
            self.keypoint_history.append(keypoints)
            self.bbox_history.append(boxes)
            self.score_history.append(scores)
        else:
            # No detections in this frame
            self.keypoint_history.append(np.array([]))
            self.bbox_history.append(np.array([]))
            self.score_history.append(np.array([]))
    
    def get_filtered_detections(self):
        """Get temporally filtered detections"""
        if len(self.keypoint_history) < self.min_detections:
            return None
        
        # Simple approach: average positions of consistent detections
        all_keypoints = []
        all_boxes = []
        all_scores = []
        
        # Collect all valid detections from recent frames
        for kps, boxes, scores in zip(self.keypoint_history, self.bbox_history, self.score_history):
            if len(kps) > 0:
                all_keypoints.extend(kps)
                all_boxes.extend(boxes)
                all_scores.extend(scores)
        
        if not all_keypoints:
            return None
        
        all_keypoints = np.array(all_keypoints)
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        
        # Cluster nearby detections and average them
        filtered_keypoints, filtered_boxes, filtered_scores = self._cluster_and_average(
            all_keypoints, all_boxes, all_scores
        )
        
        return {
            'keypoints': filtered_keypoints,
            'boxes': filtered_boxes,
            'scores': filtered_scores
        }
    
    def _cluster_and_average(self, keypoints, boxes, scores, distance_threshold=50):
        """Cluster nearby detections and return averaged results"""
        if len(keypoints) == 0:
            return [], [], []
        
        # Simple clustering based on keypoint distance
        clusters = []
        used = np.zeros(len(keypoints), dtype=bool)
        
        for i, kp in enumerate(keypoints):
            if used[i]:
                continue
                
            # Start new cluster
            cluster_kps = [kp]
            cluster_boxes = [boxes[i]]
            cluster_scores = [scores[i]]
            used[i] = True
            
            # Find nearby keypoints
            kp_center = kp[0][:2]  # Use first keypoint's x,y
            for j, other_kp in enumerate(keypoints):
                if used[j]:
                    continue
                    
                other_center = other_kp[0][:2]
                distance = np.linalg.norm(kp_center - other_center)
                
                if distance < distance_threshold:
                    cluster_kps.append(other_kp)
                    cluster_boxes.append(boxes[j])
                    cluster_scores.append(scores[j])
                    used[j] = True
            
            clusters.append((cluster_kps, cluster_boxes, cluster_scores))
        
        # Average each cluster
        final_keypoints = []
        final_boxes = []
        final_scores = []
        
        for cluster_kps, cluster_boxes, cluster_scores in clusters:
            # Average keypoints
            avg_kp = np.mean(cluster_kps, axis=0)
            avg_box = np.mean(cluster_boxes, axis=0)
            avg_score = np.mean(cluster_scores)
            
            final_keypoints.append(avg_kp)
            final_boxes.append(avg_box)
            final_scores.append(avg_score)
        
        return final_keypoints, final_boxes, final_scores


class AdvancedKalmanTracker:
    def __init__(self,
                fps=30,
                max_objects=10,
                num_keypoints_per_object=1,
                association_threshold=100.0,
                pn_scale=0.1,
                base_obs_noise=1.0,
                conf_noise_factor=0.5,
                min_confidence=0.1,
                disagreement_weight=0.5
                ):
        """
        Advanced Kalman tracker using the KeypointKalmanFilter
        
        Args:
            fps: Video frame rate for proper dt calculation
            max_objects: Maximum number of objects to track simultaneously
            num_keypoints_per_object: Number of keypoints per tracked object
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.max_objects = max_objects
        self.num_keypoints_per_object = num_keypoints_per_object
        self.object_trackers = {}  # Dict of object_id -> KeypointKalmanFilter
        self.object_metadata = {}  # Dict of object_id -> {boxes, scores, age, last_seen, keypoint_shape}
        self.next_id = 0
        self.association_threshold = association_threshold  # pixels - increased for better association
        self.max_missed_frames = 15  # Allow more missed frames before deletion
        self.min_track_age = 2  # Minimum age before returning track
        self.pn_scale = pn_scale  # Process noise scaling for stability
        self.base_obs_noise = base_obs_noise
        self.conf_noise_factor = conf_noise_factor  # Factor to scale observation noise based on confidence
        self.min_confidence = min_confidence  # Minimum confidence to avoid zero division
        self.disagreement_weight = disagreement_weight
        
    def update(self, detections, conf_threshold=0.5):
        """Update all tracked objects with new detections"""
        # Filter by confidence
        scores = detections['scores'].cpu().numpy()
        high_conf_mask = scores > conf_threshold
        
        if not np.any(high_conf_mask):
            # No detections - predict for all existing trackers
            self._predict_all_trackers()
            self._age_all_trackers()
            return self._get_stable_tracks()
        
        keypoints = detections['keypoints'][high_conf_mask].cpu().numpy()
        boxes = detections['boxes'][high_conf_mask].cpu().numpy()
        scores = scores[high_conf_mask]
        
        # Prepare detection data for advanced association
        detection_data = []
        for i, (kp, box, score) in enumerate(zip(keypoints, boxes, scores)):
            # Extract all keypoints positions and confidences
            kp_positions = []
            kp_confidences = []
            
            for j in range(min(len(kp), self.num_keypoints_per_object)):
                if len(kp[j]) >= 3:
                    x, y, visibility = kp[j]
                    if visibility > 0:
                        kp_positions.append([x, y])
                        # Use visibility as confidence proxy, scaled with detection confidence
                        kp_confidences.append(min(visibility * score, 1.0))
                    else:
                        kp_positions.append([0.0, 0.0])
                        kp_confidences.append(0.0)
            
            # Pad if needed
            while len(kp_positions) < self.num_keypoints_per_object:
                kp_positions.append([0.0, 0.0])
                kp_confidences.append(0.0)
            
            detection_data.append({
                'keypoints_full': kp,
                'keypoints_positions': np.array(kp_positions),
                'keypoints_confidences': np.array(kp_confidences),
                'box': box,
                'score': score,
                'center': kp_positions[0] if kp_positions else [0.0, 0.0]  # Use first keypoint as center
            })
        
        # Enhanced association using velocity prediction
        associations = self._enhanced_associate_detections(detection_data)
        
        # Update existing trackers
        updated_tracker_ids = set()
        for det_idx, tracker_id in associations.items():
            if tracker_id is not None:
                self._update_tracker_advanced(tracker_id, detection_data[det_idx])
                updated_tracker_ids.add(tracker_id)
        
        # Create new trackers for unassociated detections
        for det_idx in range(len(detection_data)):
            if det_idx not in associations or associations[det_idx] is None:
                if len(self.object_trackers) < self.max_objects:
                    self._create_new_tracker_advanced(detection_data[det_idx])
        
        # Predict and age trackers that weren't updated
        self._age_unused_trackers(updated_tracker_ids)
        
        # Remove old trackers
        self._remove_old_trackers()
        
        return self._get_stable_tracks()
        self._age_unused_trackers(updated_tracker_ids)
        
        # Remove old trackers
        self._remove_old_trackers()
        
        return self._get_stable_tracks()
    
    def _enhanced_associate_detections(self, detection_data):
        """Enhanced association using velocity predictions and multiple keypoints"""
        associations = {}
        
        if not self.object_trackers:
            return associations
        
        # Get predicted positions and velocities for all trackers
        tracker_predictions = {}
        for tracker_id, kalman_filter in self.object_trackers.items():
            predictions = []
            confidences = []
            
            for kp_idx in range(self.num_keypoints_per_object):
                if kp_idx < kalman_filter.num_keypoints:
                    # Get prediction for this keypoint
                    predicted_state, predicted_cov = kalman_filter.predict(kp_idx)
                    predicted_pos = predicted_state[:2]  # x, y
                    predicted_vel = predicted_state[2:4]  # vx, vy
                    
                    # Calculate uncertainty as confidence measure
                    uncertainty = np.sqrt(np.trace(predicted_cov[:2, :2]))
                    confidence = max(0.1, 1.0 / (1.0 + uncertainty / 50.0))  # Inverse uncertainty
                    
                    predictions.append({
                        'position': predicted_pos,
                        'velocity': predicted_vel,
                        'confidence': confidence
                    })
                else:
                    predictions.append({
                        'position': np.array([0.0, 0.0]),
                        'velocity': np.array([0.0, 0.0]),
                        'confidence': 0.0
                    })
            
            tracker_predictions[tracker_id] = predictions
        
        # Enhanced association using multiple criteria
        used_trackers = set()
        detection_scores = []
        
        # Calculate association scores for all detection-tracker pairs
        for det_idx, det_data in enumerate(detection_data):
            best_tracker = None
            best_score = float('inf')
            
            for tracker_id, predictions in tracker_predictions.items():
                if tracker_id in used_trackers:
                    continue
                
                # Calculate multi-keypoint association score
                total_score = 0.0
                valid_matches = 0
                
                for kp_idx in range(self.num_keypoints_per_object):
                    det_pos = det_data['keypoints_positions'][kp_idx]
                    det_conf = det_data['keypoints_confidences'][kp_idx]
                    
                    if det_conf > 0.1:  # Only consider confident detections
                        pred = predictions[kp_idx]
                        
                        # Position distance
                        pos_distance = np.linalg.norm(det_pos - pred['position'])
                        
                        # Velocity-informed prediction (look ahead by dt)
                        future_pos = pred['position'] + pred['velocity'] * self.dt
                        vel_distance = np.linalg.norm(det_pos - future_pos)
                        
                        # Combined score: weighted average of position and velocity-informed distance
                        combined_distance = 0.7 * pos_distance + 0.3 * vel_distance
                        
                        # Weight by confidences
                        weight = det_conf * pred['confidence']
                        total_score += combined_distance * weight
                        valid_matches += weight
                
                if valid_matches > 0:
                    avg_score = total_score / valid_matches
                    
                    # Add penalty for age to prefer newer tracks for ambiguous cases
                    age_penalty = self.object_metadata[tracker_id]['age'] * 2.0
                    final_score = avg_score + age_penalty
                    
                    if final_score < best_score and avg_score < self.association_threshold:
                        best_score = final_score
                        best_tracker = tracker_id
            
            if best_tracker is not None:
                associations[det_idx] = best_tracker
                used_trackers.add(best_tracker)
            else:
                associations[det_idx] = None
        
        return associations
    
    def _create_new_tracker_advanced(self, detection_data):
        """Create a new advanced Kalman filter for tracking"""
        tracker_id = self.next_id
        self.next_id += 1
        
        # Create Kalman filter with specified number of keypoints
        kalman_filter = KeypointKalmanFilter(
            num_keypoints=self.num_keypoints_per_object, 
            dt=self.dt,
            pn_scale=self.pn_scale,
            base_obs_noise=self.base_obs_noise,
            conf_noise_factor=self.conf_noise_factor,  # Factor to scale observation noise based on confidence
            min_confidence=self.min_confidence,  # Minimum confidence to avoid zero division
            disagreement_weight=self.disagreement_weight
        )
        
        # Initialize with first detection using multiple observations per keypoint
        frame_detections = []
        for kp_idx in range(self.num_keypoints_per_object):
            observations = []
            if kp_idx < len(detection_data['keypoints_positions']):
                pos = detection_data['keypoints_positions'][kp_idx]
                conf = detection_data['keypoints_confidences'][kp_idx]
                if conf > 0.1:  # Only add confident observations
                    observations.append((pos, conf))
            frame_detections.append(observations)
        
        # Process the initial frame to set up the tracker
        kalman_filter.process_frame(frame_detections)
        
        self.object_trackers[tracker_id] = kalman_filter
        self.object_metadata[tracker_id] = {
            'box': detection_data['box'].copy(),
            'score': detection_data['score'],
            'age': 1,
            'last_seen': 0,
            'keypoint_shape': detection_data['keypoints_full'].shape,
            'track_quality': detection_data['score']  # Track quality metric
        }
    
    def _update_tracker_advanced(self, tracker_id, detection_data):
        """Update an existing tracker with new advanced detection"""
        if tracker_id not in self.object_trackers:
            return
        
        kalman_filter = self.object_trackers[tracker_id]
        
        # Prepare observations for all keypoints
        frame_detections = []
        total_confidence = 0.0
        valid_keypoints = 0
        
        for kp_idx in range(self.num_keypoints_per_object):
            observations = []
            if kp_idx < len(detection_data['keypoints_positions']):
                pos = detection_data['keypoints_positions'][kp_idx]
                conf = detection_data['keypoints_confidences'][kp_idx]
                if conf > 0.1:  # Only add confident observations
                    observations.append((pos, conf))
                    total_confidence += conf
                    valid_keypoints += 1
            frame_detections.append(observations)
        
        # Process frame with multiple keypoint observations
        kalman_filter.process_frame(frame_detections)
        
        # Update metadata with exponential smoothing and quality tracking
        metadata = self.object_metadata[tracker_id]
        alpha = 0.2  # Slower adaptation for more stability
        
        metadata['box'] = alpha * detection_data['box'] + (1 - alpha) * metadata['box']
        metadata['score'] = alpha * detection_data['score'] + (1 - alpha) * metadata['score']
        metadata['age'] += 1
        metadata['last_seen'] = 0
        
        # Update track quality based on consistency
        avg_confidence = total_confidence / max(valid_keypoints, 1)
        quality_update = alpha * avg_confidence + (1 - alpha) * metadata['track_quality']
        metadata['track_quality'] = quality_update
    
    def _predict_all_trackers(self):
        """Run prediction step for all trackers when no detections are available"""
        for tracker_id, kalman_filter in self.object_trackers.items():
            # Process empty frame to maintain temporal consistency
            empty_detections = [[] for _ in range(kalman_filter.num_keypoints)]
            kalman_filter.process_frame(empty_detections)
    
    def _age_all_trackers(self):
        """Age all trackers when no detections are available"""
        for tracker_id in self.object_trackers:
            self.object_metadata[tracker_id]['last_seen'] += 1
    
    def _age_unused_trackers(self, updated_tracker_ids):
        """Age trackers that weren't updated this frame"""
        for tracker_id in self.object_trackers:
            if tracker_id not in updated_tracker_ids:
                self.object_metadata[tracker_id]['last_seen'] += 1
    
    def _remove_old_trackers(self):
        """Remove trackers that haven't been seen for too long"""
        to_remove = []
        for tracker_id, metadata in self.object_metadata.items():
            if metadata['last_seen'] > self.max_missed_frames:
                to_remove.append(tracker_id)
        
        for tracker_id in to_remove:
            del self.object_trackers[tracker_id]
            del self.object_metadata[tracker_id]
    
    def _get_stable_tracks(self):
        """Return stable tracked detections with enhanced filtering"""
        if not self.object_trackers:
            return None
        
        stable_tracks = {
            'keypoints': [],
            'boxes': [],
            'scores': []
        }
        
        for tracker_id, kalman_filter in self.object_trackers.items():
            metadata = self.object_metadata[tracker_id]
            
            # Enhanced stability criteria
            min_age = self.min_track_age
            max_missed = 3  # More restrictive for output
            min_quality = 0.3  # Minimum track quality threshold
            
            is_stable = (
                metadata['age'] >= min_age and 
                metadata['last_seen'] <= max_missed and
                metadata.get('track_quality', 0.0) >= min_quality
            )
            
            if is_stable:
                # Get current filtered positions for all keypoints
                empty_detections = [[] for _ in range(kalman_filter.num_keypoints)]
                fused_positions = kalman_filter.process_frame(empty_detections)
                
                if len(fused_positions) > 0:
                    # Reconstruct keypoint in original format
                    reconstructed_kp = np.zeros(metadata['keypoint_shape'])
                    
                    # Fill in all available keypoints
                    for kp_idx in range(min(len(fused_positions), len(reconstructed_kp))):
                        if len(reconstructed_kp[kp_idx]) >= 3:
                            pos = fused_positions[kp_idx]
                            reconstructed_kp[kp_idx][:2] = pos  # x, y
                            
                            # Set visibility based on track quality and uncertainty
                            uncertainty = kalman_filter.get_uncertainty()[kp_idx]
                            avg_uncertainty = np.mean(uncertainty)
                            
                            # Higher quality tracks with lower uncertainty get higher visibility
                            visibility = min(2.0, metadata['track_quality'] * (50.0 / (avg_uncertainty + 1.0)))
                            reconstructed_kp[kp_idx][2] = max(1.0, visibility)  # At least 1 for visible
                    
                    stable_tracks['keypoints'].append(reconstructed_kp)
                    stable_tracks['boxes'].append(metadata['box'])
                    
                    # Adjust score based on track quality and age
                    quality_bonus = metadata.get('track_quality', 0.5) * 0.2
                    age_bonus = min(0.1, metadata['age'] * 0.01)  # Small bonus for established tracks
                    adjusted_score = min(1.0, metadata['score'] + quality_bonus + age_bonus)
                    
                    stable_tracks['scores'].append(adjusted_score)
        
        return stable_tracks if stable_tracks['keypoints'] else None
    
    def get_track_statistics(self):
        """Get statistics about current tracks for debugging/monitoring"""
        if not self.object_trackers:
            return None
        
        stats = {
            'num_tracks': len(self.object_trackers),
            'track_ages': [],
            'track_qualities': [],
            'last_seen': [],
            'velocities': []
        }
        
        for tracker_id, kalman_filter in self.object_trackers.items():
            metadata = self.object_metadata[tracker_id]
            stats['track_ages'].append(metadata['age'])
            stats['track_qualities'].append(metadata.get('track_quality', 0.0))
            stats['last_seen'].append(metadata['last_seen'])
            
            # Get velocity estimates
            velocities = kalman_filter.get_velocity_estimates()
            if len(velocities) > 0:
                avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
                stats['velocities'].append(avg_velocity)
        
        return stats

# Keep the original simple tracker as a fallback
class KalmanKeyPointTracker:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.trackers = {}  # Dictionary to store trackers for each detection
        self.next_id = 0
        
    def update(self, detections, conf_threshold=0.5):
        """Update trackers with new detections"""
        # Filter by confidence
        scores = detections['scores'].cpu().numpy()
        high_conf_mask = scores > conf_threshold
        
        if not np.any(high_conf_mask):
            return None
        
        keypoints = detections['keypoints'][high_conf_mask].cpu().numpy()
        boxes = detections['boxes'][high_conf_mask].cpu().numpy()
        scores = scores[high_conf_mask]
        
        # Simple tracking: match to closest existing tracker or create new one
        updated_trackers = {}
        
        for kp, box, score in zip(keypoints, boxes, scores):
            # Find closest existing tracker
            best_match_id = None
            best_distance = float('inf')
            
            kp_center = kp[0][:2]  # Use first keypoint's x,y
            
            for tracker_id, tracker_data in self.trackers.items():
                last_pos = tracker_data['position']
                distance = np.linalg.norm(kp_center - last_pos)
                
                if distance < best_distance and distance < 100:  # Threshold for matching
                    best_distance = distance
                    best_match_id = tracker_id
            
            if best_match_id is not None:
                # Update existing tracker
                tracker_data = self.trackers[best_match_id]
                # Simple exponential smoothing
                alpha = 0.3
                new_pos = alpha * kp_center + (1 - alpha) * tracker_data['position']
                new_box = alpha * box + (1 - alpha) * tracker_data['box']
                
                updated_trackers[best_match_id] = {
                    'position': new_pos,
                    'keypoints': kp,
                    'box': new_box,
                    'score': score,
                    'age': tracker_data['age'] + 1
                }
            else:
                # Create new tracker
                updated_trackers[self.next_id] = {
                    'position': kp_center,
                    'keypoints': kp,
                    'box': box,
                    'score': score,
                    'age': 1
                }
                self.next_id += 1
        
        # Only keep trackers that were updated (removes lost tracks)
        self.trackers = {k: v for k, v in updated_trackers.items() if v['age'] > 3}  # Minimum age
        
        # Return stable tracked detections
        if self.trackers:
            tracked_keypoints = [t['keypoints'] for t in self.trackers.values()]
            tracked_boxes = [t['box'] for t in self.trackers.values()]
            tracked_scores = [t['score'] for t in self.trackers.values()]
            
            return {
                'keypoints': tracked_keypoints,
                'boxes': tracked_boxes,
                'scores': tracked_scores
            }
        
        return None


def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for inference"""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor (same as your training pipeline)
    img_tensor = F.to_tensor(img)
    
    return img_tensor, img


def preprocess_frame(frame):
    """Preprocess a video frame for inference"""
    # Frame is already in BGR format from cv2.VideoCapture
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = F.to_tensor(frame_rgb)
    return frame_tensor, frame_rgb


def draw_keypoints_on_frame(frame, result, conf_threshold=0.5, draw_boxes=True, method_name="Raw"):
    """Draw keypoints and boxes on a frame (modifies frame in-place)"""
    # Filter by confidence
    scores = result['scores'].cpu().numpy()
    high_conf_mask = scores > conf_threshold
    
    boxes = result['boxes'][high_conf_mask].cpu().numpy()
    keypoints = result['keypoints'][high_conf_mask].cpu().numpy()
    scores = scores[high_conf_mask]
    
    # Draw on frame (frame is in BGR format for cv2)
    for i, (box, kps, score) in enumerate(zip(boxes, keypoints, scores)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw keypoints
        for kp in kps:
            x, y, visibility = kp
            if visibility > 0:  # Only draw visible keypoints
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red circles
        
        if draw_boxes:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green boxes
            
            # Add score label
            cv2.putText(frame, f'{method_name}: {score:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def draw_filtered_keypoints_on_frame(frame, filtered_result, draw_boxes=True, method_name="Filtered"):
    """Draw filtered keypoints and boxes on a frame"""
    if filtered_result is None:
        return frame
    
    keypoints = filtered_result['keypoints']
    boxes = filtered_result['boxes']
    scores = filtered_result['scores']
    
    # Draw on frame (frame is in BGR format for cv2)
    for i, (kps, box, score) in enumerate(zip(keypoints, boxes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw keypoints
        if isinstance(kps, np.ndarray) and len(kps.shape) >= 2:
            for kp in kps:
                x, y, visibility = kp
                if visibility > 0:  # Only draw visible keypoints
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)  # Larger red circles
                    # Add a white border for better visibility
                    cv2.circle(frame, (int(x), int(y)), 8, (255, 255, 255), 2)
        
        if draw_boxes:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker green boxes
            
            # Add score label
            cv2.putText(frame, f'{method_name}: {score:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def draw_with_alpha(frame, filtered_result, draw_boxes=True, method_name="Filtered", alpha=1.0):
    """Draw filtered keypoints and boxes on a frame with alpha transparency"""
    if filtered_result is None:
        return frame
    
    keypoints = filtered_result['keypoints']
    boxes = filtered_result['boxes']
    scores = filtered_result['scores']
    
    # Create overlay for transparency effect
    overlay = frame.copy()
    
    # Draw on overlay (same formatting as draw_filtered_keypoints_on_frame)
    for i, (kps, box, score) in enumerate(zip(keypoints, boxes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw keypoints
        if isinstance(kps, np.ndarray) and len(kps.shape) >= 2:
            for kp in kps:
                x, y, visibility = kp
                if visibility > 0:  # Only draw visible keypoints
                    cv2.circle(overlay, (int(x), int(y)), 8, (0, 0, 255), -1)  # Larger red circles
                    # Add a white border for better visibility
                    cv2.circle(overlay, (int(x), int(y)), 8, (255, 255, 255), 2)
        
        if draw_boxes:
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker green boxes
            
            # Add score label
            cv2.putText(overlay, f'{method_name}: {score:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend overlay with original frame using alpha transparency
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        frame[:] = overlay
    
    return frame


def draw_advanced_with_alpha(frame, tracked_result, advanced_tracker, draw_boxes=True, method_name="AdvKalman", config=None, alpha=1.0):
    """
    Enhanced visualization for Advanced Kalman tracking with alpha transparency
    """
    if tracked_result is None or advanced_tracker is None:
        return frame
    
    # Use config parameters if provided, otherwise use defaults
    if config is None:
        config = {
            'min_velocity_threshold': 0.5,
            'max_velocity_display': 20.0,
            'velocity_scale': 3.0,
            'uncertainty_scale_min': 3,
            'uncertainty_scale_max': 30,
            'uncertainty_threshold': 1.0
        }
    
    keypoints = tracked_result['keypoints']
    boxes = tracked_result['boxes']
    scores = tracked_result['scores']
    
    # Get additional tracking information
    stats = advanced_tracker.get_track_statistics()
    
    # Create overlay for transparency effect
    overlay = frame.copy()
    
    # Draw on overlay (same formatting as draw_advanced_kalman_visualization)
    for i, (kps, box, score) in enumerate(zip(keypoints, boxes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw keypoints with enhanced visualization
        if isinstance(kps, np.ndarray) and len(kps.shape) >= 2:
            for kp_idx, kp in enumerate(kps):
                x, y, visibility = kp
                if visibility > 0:
                    # Main keypoint circle
                    cv2.circle(overlay, (int(x), int(y)), 8, (0, 0, 255), -1)  # Red center
                    cv2.circle(overlay, (int(x), int(y)), 10, (255, 255, 255), 2)  # White border
                    
                    # Get velocity and uncertainty for this keypoint if available
                    if hasattr(advanced_tracker, 'object_trackers') and len(advanced_tracker.object_trackers) > i:
                        tracker_id = list(advanced_tracker.object_trackers.keys())[i]
                        kalman_filter = advanced_tracker.object_trackers[tracker_id]
                        
                        if kp_idx < kalman_filter.num_keypoints:
                            # Get velocity estimate
                            velocities = kalman_filter.get_velocity_estimates()
                            uncertainties = kalman_filter.get_uncertainty()
                            
                            if len(velocities) > kp_idx:
                                vx, vy = velocities[kp_idx]
                                
                                # IMPROVED VELOCITY VISUALIZATION using config
                                velocity_magnitude = np.linalg.norm([vx, vy])
                                
                                # Only draw velocities above threshold
                                if velocity_magnitude > config['min_velocity_threshold']:
                                    # Clamp velocities to reasonable range
                                    if velocity_magnitude > config['max_velocity_display']:
                                        scale_factor = config['max_velocity_display'] / velocity_magnitude
                                        vx_clamped = vx * scale_factor
                                        vy_clamped = vy * scale_factor
                                    else:
                                        vx_clamped = vx
                                        vy_clamped = vy
                                    
                                    # Scale for visualization using config
                                    scale = config['velocity_scale']
                                    end_x = int(x + vx_clamped * scale)
                                    end_y = int(y + vy_clamped * scale)
                                    
                                    # Draw velocity vector with thickness based on confidence
                                    thickness = max(1, int(2 * visibility))
                                    cv2.arrowedLine(overlay, (int(x), int(y)), (end_x, end_y), 
                                                   (0, 255, 255), thickness, tipLength=0.3)  # Yellow arrow
                                    
                                    # Add velocity magnitude as text
                                    vel_text = f"{velocity_magnitude:.1f}"
                                    cv2.putText(overlay, vel_text, (int(x)+15, int(y)), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                            
                            # IMPROVED UNCERTAINTY VISUALIZATION using config
                            if len(uncertainties) > kp_idx:
                                ux, uy = uncertainties[kp_idx]
                                
                                # Draw uncertainty as ellipse (better scaling using config)
                                avg_uncertainty = np.mean([ux, uy])
                                uncertainty_scale = min(config['uncertainty_scale_max'], 
                                                      max(config['uncertainty_scale_min'], 
                                                          avg_uncertainty * 2))
                                
                                # Only draw uncertainty if it's meaningful
                                if avg_uncertainty > config['uncertainty_threshold']:
                                    cv2.ellipse(overlay, (int(x), int(y)), 
                                               (int(uncertainty_scale), int(uncertainty_scale)), 
                                               0, 0, 360, (255, 255, 0), 1)  # Cyan uncertainty
        
        if draw_boxes:
            # Enhanced bounding box with track information
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box
            
            # Add comprehensive label
            label_lines = [f'{method_name}: {score:.2f}']
            
            # Add track statistics if available
            if stats and i < len(stats['track_ages']):
                age = stats['track_ages'][i]
                quality = stats['track_qualities'][i] if i < len(stats['track_qualities']) else 0.0
                velocity = stats['velocities'][i] if i < len(stats['velocities']) else 0.0
                
                label_lines.append(f'Age: {age}, Q: {quality:.2f}')
                # Only show velocity if it's significant
                if velocity > config['min_velocity_threshold']:
                    label_lines.append(f'Vel: {velocity:.1f} px/f')
            
            # Draw multi-line label
            for j, line in enumerate(label_lines):
                y_offset = y1 - 10 - j * 15
                cv2.putText(overlay, line, (x1, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # IMPROVED LEGEND 
    legend_y = 30
    cv2.putText(overlay, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, "Red: Keypoints", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(overlay, f"Yellow: Velocity (>{config['min_velocity_threshold']}px/f)", (10, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    if processing_config['filter_method'] == 'advanced_kalman':
        cv2.putText(overlay, "Config:", (10, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"conf_threshold: {processing_config['conf_threshold']}", (10, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"max_objects: {kalman_config['max_objects']}", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"association_threshold: {kalman_config['max_objects']}", (10, legend_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"pn_scale: {kalman_config['base_obs_noise']}", (10, legend_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"base_obs_noise: {kalman_config['base_obs_noise']}", (10, legend_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"min_conf: {kalman_config['min_confidence']}", (10, legend_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


    # Blend overlay with original frame using alpha transparency
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        frame[:] = overlay
    
    return frame


# Update the process_video_with_options function to use cached predictions
def process_video_with_options(input_video_path,
                                output_video_path=None,
                                conf_threshold=0.5, 
                                draw_boxes=True,
                                skip_frames=1, 
                                filter_method="none",
                                window_size=5,
                                min_detections=3,
                                display_live=False,
                                enhanced_viz=False,
                                skip_alpha=1.0,
                                use_cache=True,
                                force_rerun_inference=False
                                ):
    """
    Process video with multiple filtering options and timing
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to save output video (if None, will auto-generate)
        conf_threshold: Confidence threshold for detections
        draw_boxes: Whether to draw bounding boxes
        skip_frames: Process every Nth frame (1 = process all frames)
        filter_method: Filtering method to use
        window_size: Window size for temporal filter
        min_detections: Minimum detections required for temporal filter
        display_live: Whether to show live preview (press 'q' to quit)
        enhanced_viz: Whether to use enhanced visualization (velocity vectors, uncertainty)
        skip_alpha: Alpha transparency for skipped frames (0.0 = transparent, 1.0 = opaque)
        use_cache: Whether to use cached predictions if available
        force_rerun_inference: Force rerun inference even if cache exists
    """
    
    # Start total timer
    total_start_time = time.time()
    
    # Validate filter method
    valid_methods = ["none", "temporal", "kalman", "advanced_kalman"]
    if filter_method not in valid_methods:
        print(f"Error: filter_method must be one of {valid_methods}")
        return
    
    model, device = None, None

    # Run inference with caching
    if use_cache:
        print("=" * 60)
        print("INFERENCE PHASE (with caching)")
        print("=" * 60)
        all_predictions = run_inference_with_cache(
            input_video_path,
            conf_threshold, 
            force_rerun=force_rerun_inference
        )
        if all_predictions is None:
            print("Error: Could not get predictions")
            return
    else:
        print("Cache disabled - running full inference")
        # get model
        all_predictions = run_inference_with_cache(
            input_video_path,
            conf_threshold, 
            force_rerun=True
            )   
    
    print("\n" + "=" * 60)
    print("FILTERING AND VISUALIZATION PHASE")
    print("=" * 60)
    
    # Open video for reading frames (for visualization)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = len(all_predictions)
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Filter method: {filter_method}")
    
    # Set up output video writer
    if output_video_path is None:
        base_name = os.path.splitext(input_video_path)[0]
        cache_suffix = "_cached" if use_cache else "_nocache"
        output_video_path = f"{base_name}_{filter_method}_keypoints{cache_suffix}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize filter based on method
    temporal_filter = None
    kalman_tracker = None
    advanced_kalman_tracker = None
    
    if filter_method == "temporal":
        temporal_filter = TemporalFilter(window_size=window_size, min_detections=min_detections)
        print(f"Temporal filtering: window_size={window_size}, min_detections={min_detections}")
    elif filter_method == "kalman":
        kalman_tracker = KalmanKeyPointTracker()
        print("Simple Kalman tracking initialized")
    elif filter_method == "advanced_kalman":
        # Determine number of keypoints per object
        num_keypoints = 1  # Default
        if all_predictions and len(all_predictions[0]['keypoints']) > 0:
            num_keypoints = len(all_predictions[0]['keypoints'][0])
        
        # Use parameters from config
        config = kalman_config
        
        advanced_kalman_tracker = AdvancedKalmanTracker(
            fps=fps/skip_frames, 
            max_objects=config['max_objects'], 
            num_keypoints_per_object=num_keypoints,
            association_threshold=config['association_threshold'],
            pn_scale=config['pn_scale'],
            base_obs_noise=config['base_obs_noise'],
            conf_noise_factor=config['conf_noise_factor'],
            min_confidence=config['min_confidence'],
            disagreement_weight=config['disagreement_weight']
        )
        print(f"Advanced Kalman tracking initialized (keypoints per object: {num_keypoints})")
        print(f"  Parameters: pn_scale={config['pn_scale']}, base_obs_noise={config['base_obs_noise']}")
    
    frame_count = 0
    processed_count = 0
    
    # Timing variables (only for filtering now)
    filtering_times = []
    total_processing_times = []
    
    # Store last processed result for skipped frames
    last_tracked_result = None
    last_filter_method_used = "none"
    
    print(f"Processing video... Output will be saved to: {output_video_path}")
    print("-" * 60)
    
    try:
        for prediction_data in all_predictions:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # Skip frames if specified
            if frame_count % skip_frames != 0:
                # Draw previous positions on skipped frames if we have them
                if last_tracked_result is not None and skip_frames > 1:
                    if filter_method == "advanced_kalman" and advanced_kalman_tracker is not None:
                        if enhanced_viz:
                            annotated_frame = draw_advanced_with_alpha(
                                frame.copy(), last_tracked_result, advanced_kalman_tracker,
                                draw_boxes, f"Prev-{last_filter_method_used}", 
                                viz_config, alpha=skip_alpha
                            )
                        else:
                            annotated_frame = draw_with_alpha(
                                frame.copy(), last_tracked_result, 
                                draw_boxes, f"Prev-{last_filter_method_used}", 
                                alpha=skip_alpha
                            )
                    else:
                        annotated_frame = draw_with_alpha(
                            frame.copy(), last_tracked_result, 
                            draw_boxes, f"Prev-{last_filter_method_used}", 
                            alpha=skip_alpha
                        )
                    out.write(annotated_frame)
                else:
                    out.write(frame)
                frame_count += 1
                continue
            
            # Convert prediction data back to tensor format for compatibility
            result = {
                'keypoints': torch.tensor(prediction_data['keypoints']).to(device),
                'boxes': torch.tensor(prediction_data['boxes']).to(device),
                'scores': torch.tensor(prediction_data['scores']).to(device)
            }
            
            # Apply filtering with timing
            filtering_start = time.time()
            
            if filter_method == "none":
                # No filtering - use raw detections
                annotated_frame = draw_keypoints_on_frame(frame.copy(), result, 
                                                        conf_threshold, draw_boxes, "Raw")
                last_tracked_result = {
                    'keypoints': prediction_data['keypoints'],
                    'boxes': prediction_data['boxes'],
                    'scores': prediction_data['scores']
                } if len(prediction_data['keypoints']) > 0 else None
                last_filter_method_used = "Raw"
            
            elif filter_method == "temporal":
                # Temporal filtering
                temporal_filter.add_frame(result, conf_threshold)
                filtered_result = temporal_filter.get_filtered_detections()
                
                if filtered_result is not None:
                    annotated_frame = draw_filtered_keypoints_on_frame(frame.copy(), filtered_result, 
                                                                     draw_boxes, "Temporal")
                    last_tracked_result = filtered_result
                    last_filter_method_used = "Temporal"
                else:
                    annotated_frame = draw_keypoints_on_frame(frame.copy(), result, 
                                                            conf_threshold, draw_boxes, "Raw")
                    last_tracked_result = {
                        'keypoints': prediction_data['keypoints'],
                        'boxes': prediction_data['boxes'],
                        'scores': prediction_data['scores']
                    } if len(prediction_data['keypoints']) > 0 else None
                    last_filter_method_used = "Raw"
            
            elif filter_method == "kalman":
                # Simple Kalman tracking
                tracked_result = kalman_tracker.update(result, conf_threshold)
                
                if tracked_result is not None:
                    annotated_frame = draw_filtered_keypoints_on_frame(frame.copy(), tracked_result, 
                                                                     draw_boxes, "Kalman")
                    last_tracked_result = tracked_result
                    last_filter_method_used = "Kalman"
                else:
                    annotated_frame = draw_keypoints_on_frame(frame.copy(), result, 
                                                            conf_threshold, draw_boxes, "Raw")
                    last_tracked_result = {
                        'keypoints': prediction_data['keypoints'],
                        'boxes': prediction_data['boxes'],
                        'scores': prediction_data['scores']
                    } if len(prediction_data['keypoints']) > 0 else None
                    last_filter_method_used = "Raw"
            
            elif filter_method == "advanced_kalman":
                # Advanced Kalman tracking
                tracked_result = advanced_kalman_tracker.update(result, conf_threshold)
                
                if tracked_result is not None:
                    if enhanced_viz:
                        annotated_frame = draw_advanced_with_alpha(
                            frame.copy(), tracked_result, advanced_kalman_tracker, 
                            draw_boxes, "AdvKalman", viz_config, alpha=1.0
                        )
                    else:
                        annotated_frame = draw_filtered_keypoints_on_frame(frame.copy(), tracked_result, 
                                                                         draw_boxes, "AdvKalman")
                    last_tracked_result = tracked_result
                    last_filter_method_used = "AdvKalman"
                    
                    # Print stats occasionally
                    if hasattr(advanced_kalman_tracker, 'get_track_statistics') and processed_count % 60 == 0:
                        stats = advanced_kalman_tracker.get_track_statistics()
                        if stats:
                            print(f"  Track Stats: {stats['num_tracks']} tracks, "
                                  f"avg quality: {np.mean(stats['track_qualities']):.2f}, "
                                  f"avg velocity: {np.mean(stats['velocities']):.1f} px/frame" 
                                  if stats['velocities'] else "avg velocity: 0.0 px/frame")
                else:
                    annotated_frame = draw_keypoints_on_frame(frame.copy(), result, 
                                                            conf_threshold, draw_boxes, "Raw")
                    last_tracked_result = {
                        'keypoints': prediction_data['keypoints'],
                        'boxes': prediction_data['boxes'],
                        'scores': prediction_data['scores']
                    } if len(prediction_data['keypoints']) > 0 else None
                    last_filter_method_used = "Raw"
            
            filtering_time = time.time() - filtering_start
            filtering_times.append(filtering_time)
            
            # Write to output video
            out.write(annotated_frame)
            
            # Display live preview if requested
            if display_live:
                display_frame = annotated_frame
                if width > 1280:
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(annotated_frame, (new_width, new_height))
                
                cv2.imshow(f'Keypoint Detection - {filter_method.capitalize()}', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_processing_time = time.time() - frame_start_time
            total_processing_times.append(frame_processing_time)
            
            processed_count += 1
            if processed_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_filtering = np.mean(filtering_times[-30:]) * 1000
                avg_total = np.mean(total_processing_times[-30:]) * 1000
                fps_current = 30 / np.sum(total_processing_times[-30:])
                
                print(f"Progress: {progress:.1f}% | "
                      f"Filtering: {avg_filtering:.1f}ms | "
                      f"Total: {avg_total:.1f}ms | "
                      f"FPS: {fps_current:.1f}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        # Clean up
        cap.release()
        out.release()
        
        if display_live:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        # Calculate and print timing statistics
        total_time = time.time() - total_start_time
        
        if filtering_times:
            avg_filtering = np.mean(filtering_times) * 1000
            avg_total_frame = np.mean(total_processing_times) * 1000
            effective_fps = processed_count / np.sum(total_processing_times)
            
            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE - TIMING STATISTICS")
            print("=" * 60)
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Frames processed: {processed_count}/{total_frames}")
            print(f"Original video FPS: {fps}")
            print(f"Processing FPS (filtering only): {effective_fps:.1f}")
            print(f"Speed ratio: {effective_fps/fps:.2f}x")
            print()
            print("FILTERING TIMING:")
            print(f"  Average: {avg_filtering:.1f} ms")
            print()
            print("TOTAL FRAME PROCESSING:")
            print(f"  Average: {avg_total_frame:.1f} ms per frame")
            print()
            print(f"Cache used: {use_cache}")
            print(f"Output saved to: {output_video_path}")
        else:
            print(f"No frames processed. Total time: {total_time:.2f} seconds")



def visualize_keypoint_rcnn_results(image_path, result, conf_threshold=0.5, draw_info=False):
    """Visualize Keypoint R-CNN results on a single image"""
    # Load image for display
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Filter by confidence
    scores = result['scores'].cpu().numpy()
    high_conf_mask = scores > conf_threshold
    
    boxes = result['boxes'][high_conf_mask].cpu().numpy()
    keypoints = result['keypoints'][high_conf_mask].cpu().numpy()
    scores = scores[high_conf_mask]
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    
    # Draw bounding boxes and keypoints
    for i, (box, kps, score) in enumerate(zip(boxes, keypoints, scores)):
        x1, y1, x2, y2 = box
        
        # Draw keypoints
        for kp in kps:
            x, y, visibility = kp
            if visibility > 0:  # Only draw visible keypoints
                plt.plot(x, y, 'ro', markersize=3)

        if draw_info:
            # Draw bounding box
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
                        
            # Add score label
            plt.text(x1, y1-10, f'{score:.2f}', 
                    bbox=dict(facecolor='red', alpha=0.0), fontsize=8, color='white')
    
    plt.axis('off')
    plt.title(f'Keypoint R-CNN Results (Conf > {conf_threshold})')
    plt.show()


def process_single_image_with_timing(img_idx=1):
    """Process a single image with timing information"""
    print("Processing single image with timing...")
    
    img_list = glob.glob(os.path.join("test_images", "*.*"))
    
    if img_idx >= len(img_list):
        print(f"Error: Image index {img_idx} out of range (0-{len(img_list)-1})")
        return
    
    print(f"Processing image: {img_list[img_idx]}")

    # Load model
    model, device = _get_model()
    
    # Load and preprocess timing
    load_start = time.time()
    image_tensor, original_img = load_and_preprocess_image(img_list[img_idx])
    load_time = time.time() - load_start
    
    # Inference timing
    inference_start = time.time()
    with torch.no_grad():
        result = model([image_tensor.to(device)])[0]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_time = time.time() - inference_start
    
    # Visualization timing
    viz_start = time.time()
    visualize_keypoint_rcnn_results(img_list[img_idx], result, conf_threshold=0.5)
    viz_time = time.time() - viz_start
    
    total_time = load_time + inference_time + viz_time
    
    print("\n" + "=" * 50)
    print("SINGLE IMAGE TIMING RESULTS")
    print("=" * 50)
    print(f"Image loading: {load_time*1000:.1f} ms")
    print(f"Inference: {inference_time*1000:.1f} ms")
    print(f"Visualization: {viz_time*1000:.1f} ms")
    print(f"Total: {total_time*1000:.1f} ms")
    print(f"Device: {device}")
    print()
    print("Result keys:", result.keys())
    print("Number of detections:", len(result['scores']))
    if len(result['scores']) > 0:
        print("Top 5 scores:", result['scores'][:5].cpu().numpy())
    



# Add cache management functions
def clear_inference_cache(video_path=None):
    """Clear inference cache files"""
    cache_dir = "inference_cache"
    if not os.path.exists(cache_dir):
        print("No cache directory found")
        return
    
    if video_path:
        # Clear cache for specific video
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        pattern = f"{cache_dir}/{base_name}_*.pkl"
        cache_files = glob.glob(pattern)
    else:
        # Clear all cache files
        cache_files = glob.glob(f"{cache_dir}/*.pkl")
    
    removed_count = 0
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            removed_count += 1
            print(f"Removed: {cache_file}")
        except Exception as e:
            print(f"Could not remove {cache_file}: {e}")
    
    print(f"Cleared {removed_count} cache file(s)")


#
# Add cache management functions
def clear_inference_cache(video_path=None):
    """Clear inference cache files"""
    cache_dir = "inference_cache"
    if not os.path.exists(cache_dir):
        print("No cache directory found")
        return
    
    if video_path:
        # Clear cache for specific video
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        pattern = f"{cache_dir}/{base_name}_*.pkl"
        cache_files = glob.glob(pattern)
    else:
        # Clear all cache files
        cache_files = glob.glob(f"{cache_dir}/*.pkl")
    
    removed_count = 0
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            removed_count += 1
            print(f"Removed: {cache_file}")
        except Exception as e:
            print(f"Could not remove {cache_file}: {e}")
    
    print(f"Cleared {removed_count} cache file(s)")

def list_inference_cache():
    """List all cached inference files"""
    cache_dir = "inference_cache"
    if not os.path.exists(cache_dir):
        print("No cache directory found")
        return
    
    cache_files = glob.glob(f"{cache_dir}/*.pkl")
    if not cache_files:
        print("No cache files found")
        return
    
    

# Example usage and configuration section
if __name__ == "__main__":
    video_path = "IMG_9231.MOV"  # Change to your video path
    
    # =============================================================================
    # CONFIGURATION SECTION - Modify these parameters to tune performance
    # =============================================================================
    
    # KALMAN FILTER CONFIGURATION
    kalman_config = {
        # Basic tracker settings
        'max_objects': 30,              # Maximum number of objects to track simultaneously
        'association_threshold': 10.0,  # Distance threshold for associating detections to tracks (pixels)
        
        # Kalman filter noise parameters
        'pn_scale': 0.01,               # Process noise scale (lower = smoother, higher = more responsive)
        'base_obs_noise': 0.01,          # Base observation noise (lower = trust detections more) - 1.0
        'conf_noise_factor': 0.01,      # Factor to scale obs noise based on confidence (lower = trust high conf more) - 0.1
        'min_confidence': 0.5,          # Minimum confidence threshold for tracking
        'disagreement_weight': 0.3,     # Weight for disagreement in multi-keypoint association - 0.2
    }
    
    # VISUALIZATION CONFIGURATION  
    viz_config = {
        # Velocity visualization
        'min_velocity_threshold': 0.5,  # Minimum velocity to display (pixels/frame)
        'max_velocity_display': 20.0,   # Maximum velocity to display (clamp higher values)
        'velocity_scale': 1.0,          # Scale factor for velocity arrows (lower = shorter arrows)
        
        # Uncertainty visualization
        'uncertainty_scale_min': 3,     # Minimum uncertainty ellipse size
        'uncertainty_scale_max': 30,    # Maximum uncertainty ellipse size
        'uncertainty_threshold': 1.0,   # Minimum uncertainty to display ellipse
    }
    
    # VIDEO PROCESSING CONFIGURATION
    processing_config = {
        'input_video': video_path,
        'output_video': video_path + "_kalman_enhanced_v4_rcnn_v3_2_02.mp4",
        'conf_threshold': 0.4,          # Detection confidence threshold
        'draw_boxes': False,            # Whether to draw bounding boxes
        'skip_frames': 1,               # Process every Nth frame (1 = all frames)
        'filter_method': "advanced_kalman",
        'enhanced_viz': True,           # Show velocity vectors and uncertainty
        'display_live': False,          # Show live preview while processing
        'skip_alpha': 1.0,              # Alpha transparency for skipped frames (0.0-1.0)
        
        # CACHE CONFIGURATION
        'use_cache': True,              # Use cached predictions if available
        'force_rerun_inference': True,  # Force rerun inference even if cache exists

        # NMS SETTINGS
        'apply_nms': True,              # Enable NMS for double detection filtering
        # 'nms_config': nms_config,       # NMS configuration
    }

      
    # Double detection removal
    nms_config = {
        'iou_threshold': 0.9,                   # 0.5 - 0.1
        'keypoint_distance_threshold': 15.0,    # 10 - 35
        'score_threshold': 0.2,                 # 0.05 - 0.3
        'max_detections': 20,                   # 5 - 20
    }
    
        # FOR FAST MOTION / SPORTS:
    # - Higher pn_scale (0.05-0.1) for more responsiveness
    # - Lower association_threshold (30-50) for tighter tracking
    # - Higher velocity_scale (5-10) to see movement clearly
    #
    # FOR SLOW/SMOOTH MOTION:
    # - Lower pn_scale (0.001-0.01) for smoother tracking
    # - Higher association_threshold (50-100) for more flexibility
    # - Lower velocity_scale (1-3) to avoid clutter
    #
    # FOR NOISY DETECTIONS:
    # - Higher base_obs_noise (10-20) to trust detections less
    # - Lower conf_noise_factor (0.05-0.1) to rely more on confidence
    # - Higher min_confidence (0.3-0.7) to filter low-quality detections
    #
    # FOR HIGH-QUALITY DETECTIONS:
    # - Lower base_obs_noise (1-5) to trust detections more
    # - Higher conf_noise_factor (0.2-0.5) for less confidence weighting
    # - Lower min_confidence (0.1-0.3) to include more detections


    # =============================================================================


    # Uncomment these lines to manage cache:
    
    # List all cached files
    # list_inference_cache()
    
    # Clear cache for specific video
    # clear_inference_cache(video_path)
    
    # Clear all cache
    # clear_inference_cache()
    
    # =============================================================================
    # RUN PROCESSING WITH CACHING
    # =============================================================================
    
    # RUN PROCESSING WITH CONFIGURATION
    process_video_with_options(
        input_video_path=processing_config['input_video'],
        output_video_path=processing_config['output_video'],
        conf_threshold=processing_config['conf_threshold'],
        draw_boxes=processing_config['draw_boxes'],
        skip_frames=processing_config['skip_frames'],
        filter_method=processing_config['filter_method'],
        enhanced_viz=processing_config['enhanced_viz'],
        display_live=processing_config['display_live'],
        skip_alpha=processing_config['skip_alpha'],
        use_cache=processing_config['use_cache'],
        force_rerun_inference=processing_config['force_rerun_inference']
    )
    
    # ALTERNATIVE CONFIGURATIONS FOR DIFFERENT SCENARIOS:
    
    # # High-speed motion configuration
    # kalman_config_fast = {
    #     'max_objects': 10,
    #     'association_threshold': 30.0,
    #     'pn_scale': 0.05,
    #     'base_obs_noise': 3.0,
    #     'conf_noise_factor': 0.05,
    #     'min_confidence': 0.4,
    #     'disagreement_weight': 0.1,
    # }
    
    # # Smooth/stable motion configuration
    # kalman_config_smooth = {
    #     'max_objects': 30,
    #     'association_threshold': 80.0,
    #     'pn_scale': 0.002,
    #     'base_obs_noise': 8.0,
    #     'conf_noise_factor': 0.2,
    #     'min_confidence': 0.2,
    #     'disagreement_weight': 0.3,
    # }
    
    # # Noisy detection configuration
    # kalman_config_noisy = {
    #     'max_objects': 15,
    #     'association_threshold': 60.0,
    #     'pn_scale': 0.02,
    #     'base_obs_noise': 15.0,
    #     'conf_noise_factor': 0.05,
    #     'min_confidence': 0.5,
    #     'disagreement_weight': 0.1,
    # }
    
    print("\n" + "="*60)
    print("CACHING FEATURES")
    print("="*60)
    print("Kalman Filter Settings:")
    for key, value in kalman_config.items():
        print(f"  {key}: {value}")
    print("\nVisualization Settings:")
    for key, value in viz_config.items():
        print(f"  {key}: {value}")
    print("\nProcessing Settings:")
    for key, value in processing_config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Show current cache status
    print("\nCurrent cache status:")
    list_inference_cache()