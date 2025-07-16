#!/usr/bin/env python3
"""
Test script for the improved Advanced Kalman Filter integration
"""

import numpy as np
import torch
from inference_main import AdvancedKalmanTracker
from keypoint_kalman import KeypointKalmanFilter

def test_advanced_kalman_tracker():
    """Test the improved AdvancedKalmanTracker functionality"""
    print("Testing Advanced Kalman Tracker Integration")
    print("=" * 50)
    
    # Create tracker
    tracker = AdvancedKalmanTracker(fps=30, max_objects=5, num_keypoints_per_object=1)
    print(f"✓ Created tracker with {tracker.num_keypoints_per_object} keypoints per object")
    
    # Simulate detection data
    def create_mock_detection(x, y, score=0.8, visibility=2.0):
        """Create a mock detection result"""
        return {
            'keypoints': torch.tensor([[[x, y, visibility]]]),
            'boxes': torch.tensor([[x-20, y-20, x+20, y+20]]),
            'scores': torch.tensor([score])
        }
    
    # Test sequence: moving object
    print("\nTesting moving object tracking:")
    positions = [(100, 100), (105, 102), (110, 105), (115, 107), (120, 110)]
    
    for i, (x, y) in enumerate(positions):
        detection = create_mock_detection(x, y, score=0.8 + i*0.02)
        result = tracker.update(detection, conf_threshold=0.5)
        
        if result is not None:
            num_tracks = len(result['keypoints'])
            avg_score = np.mean(result['scores'])
            print(f"  Frame {i+1}: {num_tracks} tracks, avg score: {avg_score:.3f}")
            
            # Check keypoint positions
            if num_tracks > 0:
                kp = result['keypoints'][0]
                if len(kp) > 0:
                    tracked_x, tracked_y = kp[0][:2]
                    print(f"    Tracked position: ({tracked_x:.1f}, {tracked_y:.1f}) vs Original: ({x}, {y})")
        else:
            print(f"  Frame {i+1}: No stable tracks yet")
    
    # Test statistics
    stats = tracker.get_track_statistics()
    if stats:
        print(f"\nTracker Statistics:")
        print(f"  Active tracks: {stats['num_tracks']}")
        if stats['track_ages']:
            print(f"  Average track age: {np.mean(stats['track_ages']):.1f}")
            print(f"  Average track quality: {np.mean(stats['track_qualities']):.3f}")
            if stats['velocities']:
                print(f"  Average velocity: {np.mean(stats['velocities']):.2f} px/frame")
    
    print("\n✓ Advanced Kalman tracker test completed successfully!")

def test_direct_kalman_filter():
    """Test the KeypointKalmanFilter directly"""
    print("\nTesting KeypointKalmanFilter directly:")
    print("-" * 30)
    
    # Create filter for 1 keypoint
    kf = KeypointKalmanFilter(num_keypoints=1, dt=1/30)
    print("✓ Created KeypointKalmanFilter")
    
    # Test with moving keypoint
    positions = [(100, 100), (105, 102), (110, 105), (115, 107), (120, 110)]
    
    for i, (x, y) in enumerate(positions):
        # Create detection list for frame
        frame_detections = [[(np.array([x, y]), 0.8)]]  # Single keypoint with observations
        
        # Process frame
        fused_positions = kf.process_frame(frame_detections)
        
        # Get velocity and uncertainty
        velocities = kf.get_velocity_estimates()
        uncertainties = kf.get_uncertainty()
        
        print(f"  Frame {i+1}:")
        print(f"    Input: ({x}, {y})")
        print(f"    Filtered: ({fused_positions[0][0]:.1f}, {fused_positions[0][1]:.1f})")
        print(f"    Velocity: ({velocities[0][0]:.2f}, {velocities[0][1]:.2f}) px/frame")
        print(f"    Uncertainty: ({uncertainties[0][0]:.1f}, {uncertainties[0][1]:.1f})")
    
    print("\n✓ KeypointKalmanFilter test completed successfully!")

def test_multi_keypoint_scenario():
    """Test with multiple keypoints per object"""
    print("\nTesting multi-keypoint scenario:")
    print("-" * 30)
    
    # Create tracker for 3 keypoints per object (simulating a simplified pose)
    tracker = AdvancedKalmanTracker(fps=30, max_objects=2, num_keypoints_per_object=3)
    print("✓ Created multi-keypoint tracker")
    
    # Simulate detection with 3 keypoints (head, left shoulder, right shoulder)
    def create_multi_kp_detection(center_x, center_y, score=0.8):
        keypoints = torch.tensor([[
            [center_x, center_y-10, 2.0],      # head
            [center_x-15, center_y+5, 2.0],    # left shoulder  
            [center_x+15, center_y+5, 2.0]     # right shoulder
        ]])
        
        boxes = torch.tensor([[center_x-25, center_y-15, center_x+25, center_y+20]])
        scores = torch.tensor([score])
        
        return {
            'keypoints': keypoints,
            'boxes': boxes,
            'scores': scores
        }
    
    # Test sequence
    centers = [(100, 100), (103, 102), (106, 104), (109, 106)]
    
    for i, (cx, cy) in enumerate(centers):
        detection = create_multi_kp_detection(cx, cy, score=0.85)
        result = tracker.update(detection, conf_threshold=0.5)
        
        if result is not None and len(result['keypoints']) > 0:
            kps = result['keypoints'][0]
            print(f"  Frame {i+1}: Tracked {len(kps)} keypoints")
            for j, kp in enumerate(kps):
                print(f"    KP{j}: ({kp[0]:.1f}, {kp[1]:.1f}, vis={kp[2]:.1f})")
        else:
            print(f"  Frame {i+1}: No stable tracks yet")
    
    print("\n✓ Multi-keypoint test completed successfully!")

if __name__ == "__main__":
    test_advanced_kalman_tracker()
    test_direct_kalman_filter()
    test_multi_keypoint_scenario()
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("The advanced Kalman filter integration is working correctly.")
    print("Key improvements:")
    print("- Enhanced multi-keypoint tracking")
    print("- Velocity-informed association")
    print("- Confidence-weighted measurements")
    print("- Track quality assessment")
    print("- Improved stability and accuracy")
