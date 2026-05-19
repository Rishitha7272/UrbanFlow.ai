import cv2
import numpy as np
from config import DEFAULT_ROIS, COLORS, PHASES
from logger_config import logger

class Visualizer:
    def __init__(self, rois=DEFAULT_ROIS):
        self.rois = rois
        self.colors = COLORS

    def draw_all(self, frame, detections_by_lane, raw_detections, signal_colors, controller):
        """
        Draws bounding boxes, lane ROIs, traffic light status, countdown timers, 
        and an explanatory HUD on the frame.
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()
        
        # 1. Draw Lane ROIs (semi-transparent polygons)
        overlay = frame.copy()
        for lane_name, pts in self.rois.items():
            pixel_pts = np.array(
                [(int(pt[0] * w), int(pt[1] * h)) for pt in pts], 
                dtype=np.int32
            )
            
            # Determine color based on signal state
            lane_color = signal_colors.get(lane_name, 'RED')
            color_bgr = self.colors["red"]
            if lane_color == 'GREEN':
                color_bgr = self.colors["green"]
            elif lane_color == 'YELLOW':
                color_bgr = self.colors["yellow"]
                
            # Draw semi-transparent filled polygon
            cv2.fillPoly(overlay, [pixel_pts], color_bgr)
            
            # Draw border
            cv2.polylines(canvas, [pixel_pts], True, self.colors["roi"], 2)
            
            # Draw label name near the top of the polygon
            text_pos = pixel_pts[0]
            cv2.putText(
                canvas, f"{lane_name} Lane", 
                (int(text_pos[0]), int(text_pos[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text"], 2
            )
            
        # Apply semi-transparency for the ROI fills
        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
        
        # 2. Draw Vehicle Bounding Boxes
        for det in raw_detections:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class"]
            conf = det["confidence"]
            is_emergency = det["is_emergency"]
            
            # Use red/blue blinking color for emergency vehicles, orange for regular vehicles
            if is_emergency:
                # Flashing color effect based on frame index
                color = (255, 0, 0) if (cv2.getTickCount() // 10000000) % 2 == 0 else (0, 0, 255)
                label = f"EMERGENCY ({cls.upper()}) {conf:.2f}"
                thickness = 3
            else:
                color = self.colors["bbox"]
                label = f"{cls} {conf:.2f}"
                thickness = 2
                
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            
            # Draw small label box
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - lh - 4), (x1 + lw, y1), color, -1)
            cv2.putText(canvas, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)

        # 3. Draw Traffic Light Signal Panels (HUD style in upper-left corner)
        self._draw_signal_hud(canvas, signal_colors, controller)
        
        # 4. Draw AI Decision Explanation Panel (Bottom HUD)
        self._draw_explainability_hud(canvas, controller)

        return canvas

    def _draw_signal_hud(self, canvas, signal_colors, controller):
        """
        Draws traffic light indicator panels in a neat box at the top left of the screen.
        """
        panel_w = 260
        panel_h = 160
        cv2.rectangle(canvas, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (10, 10), (10 + panel_w, 10 + panel_h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(canvas, "AI TRAFFIC SIGNALS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Active Phase & Time Left
        phase_name = controller.optimizer.phases[controller.optimizer.active_phase_idx]["name"]
        timer_val = max(0, int(controller.optimizer.timer))
        state_str = controller.optimizer.current_state
        
        cv2.putText(canvas, f"Phase: {phase_name}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
        cv2.putText(canvas, f"State: {state_str} ({timer_val}s left)", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
        
        # Draw status circle lights for each lane
        y_offset = 100
        for i, (lane, color) in enumerate(signal_colors.items()):
            x_pos = 30 + (i * 55)
            # Label
            cv2.putText(canvas, lane[:5], (x_pos - 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)
            # Red light
            r_color = self.colors["red"] if color == 'RED' else (30, 30, 50)
            cv2.circle(canvas, (x_pos, y_offset + 15), 8, r_color, -1)
            # Yellow light
            y_color = self.colors["yellow"] if color == 'YELLOW' else (30, 50, 50)
            cv2.circle(canvas, (x_pos, y_offset + 30), 8, y_color, -1)
            # Green light
            g_color = self.colors["green"] if color == 'GREEN' else (30, 50, 30)
            cv2.circle(canvas, (x_pos, y_offset + 45), 8, g_color, -1)

    def _draw_explainability_hud(self, canvas, controller):
        """
        Draws an explainable AI dashboard bar at the bottom containing vehicle stats,
        congestion metric, and optimization details.
        """
        h, w = canvas.shape[:2]
        hud_h = 100
        
        # Draw semi-transparent dark HUD at bottom
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - hud_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        
        # Draw border line
        cv2.line(canvas, (0, h - hud_h), (w, h - hud_h), (100, 100, 100), 2)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 20
        
        # Left side stats: counts & congestion score
        counts_str = ", ".join([f"{k}: {v}" for k, v in controller.current_counts.items()])
        congestion_val = sum(controller.current_densities.values())
        
        cv2.putText(canvas, "LIVE VEHICLE COUNTS:", (20, h - hud_h + 25), font, font_scale, (0, 255, 255), 1)
        cv2.putText(canvas, counts_str, (20, h - hud_h + 45), font, font_scale, self.colors["text"], 1)
        cv2.putText(canvas, f"CONGESTION INDEX (PCUs): {congestion_val:.1f}", (20, h - hud_h + 75), font, font_scale, (0, 255, 255), 1)
        
        # Right side: Decision reasoning
        active_state = controller.optimizer.current_state
        timer_val = int(controller.optimizer.timer)
        
        reason = "System idle..."
        if controller.optimizer.emergency_override_active:
            reason = f"PRIORITY OVERRIDE: Emergency vehicle detected in {controller.optimizer.emergency_lane} Lane!"
            reason_color = (0, 0, 255) # Red warning
        elif controller.optimizer.check_starvation():
            reason = f"STARVATION WARNING: Adjusting timers to prevent vehicle starvation."
            reason_color = (0, 255, 255) # Yellow warning
        else:
            reason = f"OPTIMIZATION LOGIC: Phase green duration dynamically adjusted based on vehicle density."
            reason_color = (0, 255, 0) # Green text
            
        cv2.putText(canvas, "AI EXPLAINABILITY DECISION DICTIONARY:", (w // 2 - 50, h - hud_h + 25), font, font_scale, (0, 255, 255), 1)
        cv2.putText(canvas, f"Action: Current state [{active_state}] timing: {timer_val}s remaining", (w // 2 - 50, h - hud_h + 45), font, font_scale, self.colors["text"], 1)
        cv2.putText(canvas, f"Reason: {reason}", (w // 2 - 50, h - hud_h + 65), font, font_scale, reason_color, 1)
