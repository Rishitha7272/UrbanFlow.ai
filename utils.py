import cv2
import numpy as np
import os
from logger_config import logger

def generate_mock_video(output_path="sample_traffic.mp4", duration_sec=30, fps=30):
    """
    Generates a synthetic traffic simulation video showing a 4-way intersection
    with moving circles representing vehicles, including a periodic emergency vehicle.
    """
    logger.info(f"Generating mock traffic video at '{output_path}'...")
    
    width, height = 850, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    # Define lanes paths (simple linear equations or moving coordinates)
    # North-bound lane (bottom to top): x fixed at 410, y moves from 480 to 0
    # South-bound lane (top to bottom): x fixed at 440, y moves from 0 to 480
    # East-bound lane (left to right): y fixed at 250, x moves from 0 to 850
    # West-bound lane (right to left): y fixed at 220, x moves from 850 to 0
    
    # Active vehicles pool: each vehicle is a dict with {"type": "car"/"bus"/"motorcycle"/"ambulance", "pos": [x, y], "speed": dy/dx}
    vehicles = []
    
    # Pre-populate some vehicles
    for i in range(10):
        lane = np.random.choice(["N", "S", "E", "W"])
        progress = np.random.uniform(0, 1)
        v_type = np.random.choice(["car", "motorcycle", "bus", "truck"], p=[0.6, 0.2, 0.1, 0.1])
        
        if lane == "N":
            vehicles.append({"type": v_type, "pos": [410, int(height * progress)], "speed": -np.random.randint(2, 5), "lane": "N"})
        elif lane == "S":
            vehicles.append({"type": v_type, "pos": [440, int(height * progress)], "speed": np.random.randint(2, 5), "lane": "S"})
        elif lane == "E":
            vehicles.append({"type": v_type, "pos": [int(width * progress), 250], "speed": np.random.randint(3, 6), "lane": "E"})
        elif lane == "W":
            vehicles.append({"type": v_type, "pos": [int(width * progress), 220], "speed": -np.random.randint(3, 6), "lane": "W"})

    for frame_idx in range(total_frames):
        # Draw background: dark gray canvas
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40
        
        # Draw roads: 4-way intersection
        # North-South road
        cv2.rectangle(frame, (380, 0), (470, height), (70, 70, 70), -1)
        # East-West road
        cv2.rectangle(frame, (0, 190), (width, 280), (70, 70, 70), -1)
        
        # Draw lane dividing dashed lines (yellow)
        # N-S divider
        cv2.line(frame, (425, 0), (425, height), (0, 255, 255), 2)
        # E-W divider
        cv2.line(frame, (0, 235), (width, 235), (0, 255, 255), 2)
        
        # Periodic generation of new vehicles
        if frame_idx % 20 == 0:
            lane = np.random.choice(["N", "S", "E", "W"])
            # Every 300 frames (10 seconds), spawn an emergency vehicle (ambulance)
            is_amb = (frame_idx > 0 and frame_idx % 250 == 0)
            v_type = "ambulance" if is_amb else np.random.choice(["car", "motorcycle", "bus", "truck"], p=[0.6, 0.2, 0.1, 0.1])
            
            if lane == "N":
                vehicles.append({"type": v_type, "pos": [410, height], "speed": -np.random.randint(3, 5), "lane": "N"})
            elif lane == "S":
                vehicles.append({"type": v_type, "pos": [440, 0], "speed": np.random.randint(3, 5), "lane": "S"})
            elif lane == "E":
                vehicles.append({"type": v_type, "pos": [0, 250], "speed": np.random.randint(4, 7), "lane": "E"})
            elif lane == "W":
                vehicles.append({"type": v_type, "pos": [width, 220], "speed": -np.random.randint(4, 7), "lane": "W"})
                
        # Draw and update vehicles
        active_vehicles = []
        for v in vehicles:
            # Update position
            if v["lane"] in ["N", "S"]:
                v["pos"][1] += v["speed"]
            else:
                v["pos"][0] += v["speed"]
                
            # Keep vehicle if it's within screen boundaries
            if 0 <= v["pos"][0] <= width and 0 <= v["pos"][1] <= height:
                active_vehicles.append(v)
                
                # Draw vehicle body (representing as rectangles for detector compatibility if needed, or simple circles)
                # To make it look like a vehicle:
                x, y = v["pos"]
                if v["type"] == "ambulance":
                    # White body
                    cv2.rectangle(frame, (x-12, y-12), (x+12, y+12), (240, 240, 240), -1)
                    # Red cross
                    cv2.line(frame, (x, y-6), (x, y+6), (0, 0, 255), 2)
                    cv2.line(frame, (x-6, y), (x+6, y), (0, 0, 255), 2)
                    # Blinking emergency bar (flashing red and blue lights)
                    light_color = (255, 0, 0) if (frame_idx // 5) % 2 == 0 else (0, 0, 255)
                    cv2.rectangle(frame, (x-8, y-12), (x+8, y-8), light_color, -1)
                elif v["type"] == "bus":
                    # Long yellow rectangle
                    cv2.rectangle(frame, (x-15, y-25), (x+15, y+25), (0, 200, 220), -1)
                elif v["type"] == "truck":
                    # Large blue rectangle
                    cv2.rectangle(frame, (x-14, y-22), (x+14, y+22), (200, 100, 50), -1)
                elif v["type"] == "motorcycle":
                    # Small orange circle
                    cv2.circle(frame, (x, y), 6, (0, 100, 250), -1)
                else: # car
                    # Medium green/red/gray rectangle
                    cv2.rectangle(frame, (x-10, y-15), (x+10, y+15), (120, 120, 120), -1)
                    
        vehicles = active_vehicles
        
        # Write frame to video
        out.write(frame)
        
    out.release()
    logger.info(f"Mock video generated successfully with {total_frames} frames.")
    return output_path
