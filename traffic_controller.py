import time
import os
import cv2
import pandas as pd
from detector import VehicleDetector
from signal_logic import TrafficSignalOptimizer
from logger_config import logger
from config import DEFAULT_ROIS

class TrafficController:
    def __init__(self, model_path="yolov8n.pt", confidence=0.3):
        """
        Initializes the overall Traffic System Coordinator.
        """
        self.detector = VehicleDetector(model_path=model_path, confidence=confidence)
        self.optimizer = TrafficSignalOptimizer()
        self.log_history = []
        
        # Current active metrics
        self.current_counts = {}
        self.current_densities = {}
        self.emergency_lane = None
        self.emergency_detected = False
        
    def process_frame(self, frame, rois=DEFAULT_ROIS, frame_idx=0, fps=30.0):
        """
        Processes a single video frame. Computes vehicle count, densities,
        runs the signal optimization cycle, and updates states.
        
        Args:
            frame: cv2 image frame.
            rois: dict of lane names to list of normalized points.
            frame_idx: current index of the frame.
            fps: video frames per second.
            
        Returns:
            detections_by_lane: details of vehicles in each lane.
            raw_detections: all bbox and class details.
            signal_colors: dict of lane colors ('RED', 'YELLOW', 'GREEN').
        """
        # Run vehicle detection
        detections_by_lane, raw_detections, emergency_detected = self.detector.detect_vehicles(frame, rois)
        
        # Extract vehicle counts and densities
        self.current_counts = {lane: det["count"] for lane, det in detections_by_lane.items()}
        self.current_densities = {lane: det["density"] for lane, det in detections_by_lane.items()}
        self.emergency_detected = emergency_detected
        
        # Identify which lane has the emergency vehicle (first found)
        self.emergency_lane = None
        if emergency_detected:
            for lane, det in detections_by_lane.items():
                if any(v["is_emergency"] for v in det["vehicles"]):
                    self.emergency_lane = lane
                    break
        
        # Run signal logic updates once per simulated second
        # (e.g. if video runs at 30 fps, we run logic update once every 30 frames)
        simulated_second_passed = (frame_idx % int(fps) == 0) if fps > 0 else True
        if simulated_second_passed:
            self.optimizer.handle_state_transitions(
                self.current_densities, 
                self.emergency_lane, 
                dt=1.0
            )
            
            # Log metrics to traffic_system.log
            self._log_metrics(frame_idx)
            
        # Get active signal colors for each lane
        signal_colors = self.optimizer.get_signal_colors()
        
        return detections_by_lane, raw_detections, signal_colors

    def _log_metrics(self, frame_idx):
        """
        Helper method to log metrics in a structured format and record to log history list.
        """
        # Calculate total intersection congestion score (sum of densities)
        total_congestion = sum(self.current_densities.values())
        
        # Explain decision making for AI Explainability
        active_phase_name = self.optimizer.phases[self.optimizer.active_phase_idx]["name"]
        reason = f"Allocated green time based on lane density. State={self.optimizer.current_state}, Timer={self.optimizer.timer}s"
        if self.optimizer.emergency_override_active:
            reason = f"EMERGENCY OVERRIDE active for lane '{self.optimizer.emergency_lane}'"
        elif self.optimizer.check_starvation():
            reason = f"Starvation prevention override triggered"

        # Log to system file
        logger.info(
            f"Frame: {frame_idx} | Phase: {active_phase_name} | "
            f"Counts: {self.current_counts} | Congestion Score: {total_congestion:.1f} | "
            f"Emergency: {self.optimizer.emergency_override_active} | Decision Reason: {reason}"
        )
        
        # Add to history list for CSV exporting
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame_idx": frame_idx,
            "active_phase": active_phase_name,
            "state": self.optimizer.current_state,
            "timer": self.optimizer.timer,
            "total_congestion": total_congestion,
            "emergency_override": self.optimizer.emergency_override_active,
            "emergency_lane": self.optimizer.emergency_lane,
            "reason": reason
        }
        # Add per-lane counts and densities
        for lane in self.current_counts.keys():
            log_entry[f"{lane}_count"] = self.current_counts[lane]
            log_entry[f"{lane}_density"] = self.current_densities[lane]
            
        self.log_history.append(log_entry)

    def export_csv(self, filename="traffic_analytics.csv"):
        """
        Exports the logged traffic optimization data to a CSV.
        """
        if not self.log_history:
            logger.warning("No traffic logs recorded. Skipping CSV export.")
            return False
            
        try:
            df = pd.DataFrame(self.log_history)
            df.to_csv(filename, index=False)
            logger.info(f"Traffic analytics successfully exported to {filename}.")
            return True
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False
