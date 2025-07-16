import numpy as np
from typing import List, Tuple, Optional

class KeypointKalmanFilter:
    """
    Temporal Kalman Filter for fusing multiple keypoint detectors across frames.
    
    State vector: [x, y, vx, vy] for each keypoint
    - x, y: current position
    - vx, vy: velocity components
    """
    
    def __init__(
            self, num_keypoints: int,
            dt: float = 1/30,
            pn_scale: float = 0.1,
            base_obs_noise: float = 1.0,
            conf_noise_factor: float = 0.5,
            min_confidence: float = 0.1,
            disagreement_weight: float = 1.0,
):
        """
        Initialize Kalman filter for keypoint tracking.
        
        Args:
            num_keypoints: Number of keypoints to track
            dt: Time step between frames (default: 1/30 for 30fps)
        """
        self.num_keypoints = num_keypoints
        self.dt = dt
        self.state_dim = 4  # [x, y, vx, vy]
        self.base_obs_noise = base_obs_noise
        self.conf_noise_factor = conf_noise_factor  # Factor to scale observation noise based on confidence
        self.min_confidence = min_confidence  # Minimum confidence to avoid zero division
        self.disagreement_weight = disagreement_weight  # Weight for disagreement-based noise
        
        # Initialize state for each keypoint
        self.states = []  # List of state vectors for each keypoint
        self.covariances = []  # List of covariance matrices
        
        # Motion model: constant velocity
        self.F = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ]
        ])
        
        # Process noise (acceleration uncertainty)
        q = pn_scale  # Process noise scaling
        self.Q = q * np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ]
        ])
        
        # Observation model: we observe position only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Initialize filters for each keypoint
        self._initialize_filters()
    
    def _initialize_filters(self):
        """Initialize state and covariance for each keypoint."""
        for i in range(self.num_keypoints):
            # Initial state: [x, y, vx, vy] = [0, 0, 0, 0]
            initial_state = np.zeros(self.state_dim)
            
            # Initial covariance: high uncertainty
            initial_cov = np.eye(self.state_dim) * 1000
            
            self.states.append(initial_state)
            self.covariances.append(initial_cov)
    
    def predict(self, keypoint_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step for a single keypoint.
        
        Args:
            keypoint_idx: Index of the keypoint to predict
            
        Returns:
            Predicted state and covariance
        """
        # Predict next state: x_k|k-1 = F * x_k-1|k-1
        predicted_state = self.F @ self.states[keypoint_idx]
        
        # Predict covariance: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        predicted_cov = self.F @ self.covariances[keypoint_idx] @ self.F.T + self.Q
        
        return predicted_state, predicted_cov
    
    def update(self, keypoint_idx: int, observations: List[Tuple[np.ndarray, float]], 
               predicted_state: np.ndarray, predicted_cov: np.ndarray):
        """
        Update step: fuse multiple detector observations.
        
        Args:
            keypoint_idx: Index of the keypoint to update
            observations: List of (detection, confidence) pairs
            predicted_state: Predicted state from prediction step
            predicted_cov: Predicted covariance from prediction step
        """
        if not observations:
            # No observations, use prediction as final estimate
            self.states[keypoint_idx] = predicted_state
            self.covariances[keypoint_idx] = predicted_cov
            return
        
        # Fuse multiple observations using weighted average
        fused_observation, observation_noise = self._fuse_observations(observations)
        
        # Kalman gain: K = P_k|k-1 * H^T * (H * P_k|k-1 * H^T + R)^-1
        S = self.H @ predicted_cov @ self.H.T + observation_noise
        K = predicted_cov @ self.H.T @ np.linalg.inv(S)
        
        # Innovation: y = z - H * x_k|k-1
        innovation = fused_observation - self.H @ predicted_state
        
        # Update state: x_k|k = x_k|k-1 + K * y
        self.states[keypoint_idx] = predicted_state + K @ innovation
        
        # Update covariance: P_k|k = (I - K * H) * P_k|k-1
        I = np.eye(self.state_dim)
        self.covariances[keypoint_idx] = (I - K @ self.H) @ predicted_cov
    
    def _fuse_observations(self, observations: List[Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse multiple detector observations using confidence weighting.
        
        Args:
            observations: List of (detection [x,y], confidence) pairs
            
        Returns:
            Fused observation and observation noise matrix
        """

        if len(observations) == 1:
            detection, confidence = observations[0]
            # TUNABLE: Single observation confidence noise
            confidence_noise_std = self.base_obs_noise * self.conf_noise_factor / max(confidence, self.min_confidence)
            observation_noise = np.eye(2) * confidence_noise_std**2
            return detection, observation_noise
        

        # Weight observations by confidence
        total_weight = 0
        fused_obs = np.zeros(2)
        
        for detection, confidence in observations:
            weight = max(confidence, self.min_confidence)  # Apply minimum confidence
            fused_obs += weight * detection
            total_weight += weight
        
        fused_obs /= total_weight
        
        # Compute observation noise based on:
        # 1. Average confidence
        # 2. Disagreement between detectors
        avg_confidence = total_weight / len(observations)
        
        # Measure disagreement
        disagreement = 0
        for detection, confidence in observations:
            disagreement += confidence * np.linalg.norm(detection - fused_obs)**2
        disagreement /= total_weight
        
        # TUNABLE: Observation noise calculation
        # 1. Confidence-based noise: lower confidence = higher noise
        confidence_noise = self.base_obs_noise * self.conf_noise_factor / max(avg_confidence, self.min_confidence)
        
        # 2. Disagreement-based noise: more disagreement = higher noise
        disagreement_noise = self.disagreement_weight * np.sqrt(disagreement + 1e-6)
        
        # 3. Total observation noise standard deviation
        total_noise_std = confidence_noise + disagreement_noise
        
        # 4. Convert to covariance matrix (noise squared)
        observation_noise = np.eye(2) * total_noise_std**2
        
        
        return fused_obs, observation_noise
    
    def process_frame(self, frame_detections: List[List[Tuple[np.ndarray, float]]]) -> np.ndarray:
        """
        Process a single frame with detections from multiple systems.
        
        Args:
            frame_detections: List of detections for each keypoint
                            Each element is a list of (detection, confidence) pairs
                            
        Returns:
            Fused keypoint positions [num_keypoints, 2]
        """
        fused_keypoints = np.zeros((self.num_keypoints, 2))
        
        for keypoint_idx in range(self.num_keypoints):
            # Prediction step
            predicted_state, predicted_cov = self.predict(keypoint_idx)
            
            # Update step with observations
            observations = frame_detections[keypoint_idx] if keypoint_idx < len(frame_detections) else []
            self.update(keypoint_idx, observations, predicted_state, predicted_cov)
            
            # Extract position from state [x, y, vx, vy]
            fused_keypoints[keypoint_idx] = self.states[keypoint_idx][:2]
        
        return fused_keypoints
    
    def get_velocity_estimates(self) -> np.ndarray:
        """Get current velocity estimates for all keypoints."""
        velocities = np.zeros((self.num_keypoints, 2))
        for i in range(self.num_keypoints):
            velocities[i] = self.states[i][2:4]  # Extract vx, vy
        return velocities
    
    def get_uncertainty(self) -> np.ndarray:
        """Get position uncertainty (standard deviation) for all keypoints."""
        uncertainties = np.zeros((self.num_keypoints, 2))
        for i in range(self.num_keypoints):
            # Extract position covariance and compute standard deviation
            pos_cov = self.covariances[i][:2, :2]
            uncertainties[i] = np.sqrt(np.diag(pos_cov))
        return uncertainties

# Example usage demonstrating frame processing
def example_keypoint_fusion():
    """Example of using the Kalman filter for keypoint fusion across frames."""
    
    # Initialize filter for 17 keypoints (human pose)
    kf = KeypointKalmanFilter(num_keypoints=17, dt=1/30)
    
    # Simulate 10 frames of processing
    print("Processing frames with multiple detector fusion:")
    print("=" * 50)
    
    for frame_idx in range(10):
        # Simulate detections from 3 different systems for first 3 keypoints
        frame_detections = []
        
        for keypoint_idx in range(3):  # Only show first 3 keypoints
            # Simulate detections with different noise and confidence
            true_pos = np.array([100 + frame_idx * 2, 200 + frame_idx * 1])  # Moving keypoint
            
            detections = []
            # Detector 1: High confidence, low noise
            det1 = true_pos + np.random.normal(0, 1, 2)
            detections.append((det1, 0.9))
            
            # Detector 2: Medium confidence, medium noise
            det2 = true_pos + np.random.normal(0, 3, 2)
            detections.append((det2, 0.7))
            
            # Detector 3: Lower confidence, higher noise
            det3 = true_pos + np.random.normal(0, 5, 2)
            detections.append((det3, 0.5))
            
            frame_detections.append(detections)
        
        # Add empty detections for remaining keypoints
        for keypoint_idx in range(3, 17):
            frame_detections.append([])
        
        # Process frame
        fused_keypoints = kf.process_frame(frame_detections)
        
        # Display results for first 3 keypoints
        print(f"Frame {frame_idx}:")
        for i in range(3):
            pos = fused_keypoints[i]
            uncertainty = kf.get_uncertainty()[i]
            velocity = kf.get_velocity_estimates()[i]
            
            print(f"  Keypoint {i}: pos=({pos[0]:.1f}, {pos[1]:.1f}), "
                  f"vel=({velocity[0]:.1f}, {velocity[1]:.1f}), "
                  f"unc=({uncertainty[0]:.1f}, {uncertainty[1]:.1f})")
        print()

if __name__ == "__main__":
    example_keypoint_fusion()