import cv2
import numpy as np
from config import YOLO_MODEL, CONFIDENCE_THRESHOLD, VEHICLE_CLASSES, PCU_WEIGHTS, DEFAULT_ROIS
from logger_config import logger

class VehicleDetector:
    def __init__(self, model_path=YOLO_MODEL, confidence=CONFIDENCE_THRESHOLD):
        """
        Initializes the YOLOv8 detector. Falls back to mock mode if model_path is 'mock'
        or if loading the model errors out.
        """
        self.confidence = confidence
        self.use_mock = False
        
        if model_path.lower() == "mock":
            logger.info("Initializing VehicleDetector in Mock/Simulation mode...")
            self.use_mock = True
            return
            
        try:
            logger.info(f"Loading YOLO model from {model_path}...")
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}. Falling back to Mock/Simulation mode.")
            self.use_mock = True

    def detect_vehicles(self, frame, rois=DEFAULT_ROIS):
        """
        Processes a single frame and returns detected vehicles grouped by lane.
        
        Args:
            frame: cv2 image frame.
            rois: dict of lane names to list of normalized (x,y) points.
            
        Returns:
            detections_by_lane: dict containing vehicle details for each lane.
            raw_detections: list of all detections (bbox, conf, class) for rendering.
            emergency_detected: bool, true if any emergency vehicle is present in the intersection.
        """
        h, w = frame.shape[:2]
        
        # Convert normalized ROIs to pixel coordinates
        pixel_rois = {}
        for lane_name, pts in rois.items():
            pixel_rois[lane_name] = np.array(
                [(int(pt[0] * w), int(pt[1] * h)) for pt in pts], 
                dtype=np.int32
            )

        if self.use_mock:
            return self._detect_mock(frame, pixel_rois)

        # Run YOLO inference
        results = self.model(frame, verbose=False)[0]
        
        # Initialize output structures
        detections_by_lane = {
            lane: {"vehicles": [], "count": 0, "density": 0.0}
            for lane in rois.keys()
        }
        
        raw_detections = []
        emergency_detected = False

        # Parse detection results
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Filter class and confidence
            if cls_id not in VEHICLE_CLASSES or conf < self.confidence:
                continue
                
            class_name = VEHICLE_CLASSES[cls_id]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Compute center and bottom-center points of bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            bottom_center = (center_x, y2)
            
            # Check if this vehicle is emergency vehicle (e.g. ambulance or police)
            is_emergency = self._check_emergency_vehicle(frame, (x1, y1, x2, y2), class_name)
            if is_emergency:
                emergency_detected = True
                class_name = "ambulance"  # Upgrade to ambulance for density calculations
            
            pcu_val = PCU_WEIGHTS.get(class_name, 1.0)
            
            raw_det = {
                "bbox": (x1, y1, x2, y2),
                "class": class_name,
                "confidence": conf,
                "pcu": pcu_val,
                "is_emergency": is_emergency
            }
            raw_detections.append(raw_det)
            
            # Assign vehicle to matching lane ROI polygon
            assigned_lane = None
            for lane_name, poly in pixel_rois.items():
                # Use bottom-center point for ground-plane alignment check
                dist = cv2.pointPolygonTest(poly, (float(bottom_center[0]), float(bottom_center[1])), False)
                if dist >= 0:  # Inside or on the edge
                    assigned_lane = lane_name
                    break
                    
            if assigned_lane:
                detections_by_lane[assigned_lane]["vehicles"].append(raw_det)
                detections_by_lane[assigned_lane]["count"] += 1
                detections_by_lane[assigned_lane]["density"] += pcu_val

        if emergency_detected:
            logger.warning("Emergency vehicle detected in intersection! Active priority override.")

        return detections_by_lane, raw_detections, emergency_detected

    def _check_emergency_vehicle(self, frame, bbox, class_name):
        """
        Heuristic to identify emergency vehicles based on class labels and visual features 
        (e.g., flashing red/blue light patterns using HSV thresholding on the vehicle's top half).
        """
        x1, y1, x2, y2 = bbox
        
        # Emergency vehicles are usually cars, trucks, or buses
        if class_name not in ["car", "truck", "bus"]:
            return False
            
        # Focus on the top 30% of the vehicle (where emergency lights are located)
        lightbar_h = int((y2 - y1) * 0.3)
        if lightbar_h <= 0:
            return False
            
        crop_y2 = y1 + lightbar_h
        crop = frame[y1:crop_y2, x1:x2]
        
        if crop.size == 0:
            return False
            
        # Convert crop to HSV color space
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Color ranges for emergency lights (red and blue flashing strobe lights)
        # Red has two wraps in HSV space
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Blue
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        
        mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
        
        red_pixels = cv2.countNonZero(mask_r1) + cv2.countNonZero(mask_r2)
        blue_pixels = cv2.countNonZero(mask_b)
        
        total_pixels = crop.shape[0] * crop.shape[1]
        
        # Heuristic: If both red and blue pixels or a high density of red/blue is found, it's likely an emergency lightbar
        red_ratio = red_pixels / total_pixels
        blue_ratio = blue_pixels / total_pixels
        
        if (red_ratio > 0.015 and blue_ratio > 0.015) or (red_ratio > 0.04) or (blue_ratio > 0.04):
            return True
            
        return False

    def _detect_mock(self, frame, pixel_rois):
        """
        Generates simulated detections based on the intersection shape, pixel_rois,
        and current frame information. Used as a lightweight developer fallback.
        """
        detections_by_lane = {
            lane: {"vehicles": [], "count": 0, "density": 0.0}
            for lane in pixel_rois.keys()
        }
        raw_detections = []
        emergency_detected = False
        
        # We can use tick count or frame index to make it dynamic
        tick = cv2.getTickCount() // 20000000
        
        # Every ~120 ticks, trigger emergency vehicle in North lane
        has_emergency = (tick % 120 == 0) or (tick % 121 == 0)
        
        for idx, (lane, poly) in enumerate(pixel_rois.items()):
            # Calculate mock count for this lane (cycles over time)
            base_count = 2 + (idx * 2)
            offset = int(2.5 * np.sin(tick / 10.0 + idx))
            count = max(0, base_count + offset)
            
            # If emergency is active, force vehicle count in that lane
            if has_emergency and lane == "North":
                count = max(1, count)
                emergency_detected = True
            
            # Place vehicles inside the polygon bounds
            poly_center = np.mean(poly, axis=0)
            
            for v_idx in range(count):
                # Spread vehicles out from the polygon center
                offset_x = int(35 * np.cos(v_idx * 1.5))
                offset_y = int(35 * np.sin(v_idx * 1.5))
                vx = int(poly_center[0] + offset_x)
                vy = int(poly_center[1] + offset_y)
                
                # Determine vehicle type
                is_amb = (emergency_detected and lane == "North" and v_idx == 0)
                v_type = "ambulance" if is_amb else np.random.choice(["car", "motorcycle", "bus", "truck"], p=[0.7, 0.15, 0.1, 0.05])
                
                pcu_val = PCU_WEIGHTS.get(v_type, 1.0)
                
                # Create bounding box dimensions based on type
                if v_type == "ambulance":
                    bw, bh = 24, 24
                elif v_type == "bus":
                    bw, bh = 30, 50
                elif v_type == "truck":
                    bw, bh = 28, 44
                elif v_type == "motorcycle":
                    bw, bh = 12, 12
                else: # car
                    bw, bh = 20, 30
                    
                det = {
                    "bbox": (vx - bw, vy - bh, vx + bw, vy + bh),
                    "class": v_type,
                    "confidence": 0.92,
                    "pcu": pcu_val,
                    "is_emergency": is_amb
                }
                
                raw_detections.append(det)
                detections_by_lane[lane]["vehicles"].append(det)
                detections_by_lane[lane]["count"] += 1
                detections_by_lane[lane]["density"] += pcu_val
                
        return detections_by_lane, raw_detections, emergency_detected
